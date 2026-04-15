package com.traffic.violation

/**
 * ViolationDetector.kt
 *
 * 개선 사항:
 *   1. 추론 주기 30→15프레임 (0.5초마다)
 *   2. 위반 유형별 연속 감지 기준 차별화
 *   3. 정상 주행 게이트 (프레임 차분으로 차량 정지 감지)
 *   4. HSV 보정 고도화 (노란선 방향 확인)
 *   5. 브레이크등 필터 (신호위반 시 신호등 위치 검증)
 *   6. 정지선 감지 (화면 하단 수평 흰선 추적)
 */

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.ArrayDeque

data class DetectionResult(
    val violationType: String,
    val confidence: Float,
    val probabilities: FloatArray
)

class ViolationDetector(private val context: Context) {

    companion object {
        const val SEQUENCE_LENGTH    = 25
        const val IMAGE_SIZE         = 64
        const val INFERENCE_INTERVAL = 15      // 0.5초마다 추론 (기존 30→15)
        const val CONFIDENCE_THRESHOLD = 0.7f

        // 위반 유형별 연속 감지 기준
        // 신호위반: 신호등 앞 정지 상황이라 여유 있게 3회
        // 중앙선/진로변경: 빠르게 발생하므로 2회
        val CONFIRM_COUNTS = mapOf(
            "신호위반"    to 3,
            "중앙선침범"  to 2,
            "진로변경위반" to 2
        )

        val LABELS = arrayOf("신호위반", "중앙선침범", "진로변경위반")

        const val LRCN_MODEL = "lrcn_fp16.tflite"
        val YOLO_MODELS = mapOf(
            "신호위반"    to "yolo_신호위반.onnx",
            "중앙선침범"  to "yolo_중앙선침범.onnx",
            "진로변경위반" to "yolo_진로변경.onnx"
        )

        // 정상 주행 게이트: 평균 픽셀 차분이 이 값 미만이면 차량 정지로 판단
        const val MOTION_THRESHOLD = 8
        // 정지선 감지: 가로 방향 흰색 픽셀이 행 너비의 이 비율 이상이면 정지선
        const val STOP_LINE_RATIO  = 0.55f
    }

    private val frameBuffer = ArrayDeque<Bitmap>(SEQUENCE_LENGTH)

    private var frameCount       = 0
    private var consecutiveCount = 0
    private var lastDetectedType = ""

    // 정상 주행 게이트용 — 이전 프레임 그레이스케일
    private var prevGray: IntArray? = null

    // 정지선 감지용 — 이전 프레임에 정지선이 있었는지
    private var stopLineWasVisible = false

    // 매 추론마다 호출 — OverlayView 업데이트용
    var onDebugUpdate: ((DetectionDebugState) -> Unit)? = null

    private var lrcnInterpreter: Interpreter? = null
    private var ortEnvironment: OrtEnvironment? = null
    private val yoloSessions = mutableMapOf<String, OrtSession>()

    fun initialize() {
        val lrcnOptions = Interpreter.Options().apply { setNumThreads(2) }
        lrcnInterpreter = Interpreter(loadModelFile(LRCN_MODEL), lrcnOptions)

        ortEnvironment = OrtEnvironment.getEnvironment()
        YOLO_MODELS.forEach { (violationType, modelFile) ->
            val modelBytes = context.assets.open(modelFile).readBytes()
            yoloSessions[violationType] = ortEnvironment!!.createSession(modelBytes)
        }
    }

    fun processFrame(bitmap: Bitmap): DetectionResult? {
        val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        if (frameBuffer.size >= SEQUENCE_LENGTH) frameBuffer.pollFirst()?.recycle()
        frameBuffer.addLast(resized)
        frameCount++

        if (frameCount % INFERENCE_INTERVAL != 0) return null
        if (frameBuffer.size < SEQUENCE_LENGTH) return null

        // ── 정상 주행 게이트 ──────────────────────────────
        // 차량이 거의 움직이지 않으면 신호위반 외 추론 스킵
        val isMoving = isVehicleMoving(bitmap)

        // ── 정지선 감지 ───────────────────────────────────
        val stopLineNow = detectStopLine(bitmap)
        val crossedStopLine = stopLineWasVisible && !stopLineNow
        stopLineWasVisible = stopLineNow

        if (crossedStopLine) {
            Log.d("ViolationDetector", "정지선 통과 감지")
        }

        return runInference(bitmap, isMoving, crossedStopLine)
    }

    private fun runInference(
        originalFrame: Bitmap,
        isMoving: Boolean,
        crossedStopLine: Boolean
    ): DetectionResult? {

        // ── [1단계] YOLO 먼저 실행 ────────────────────────
        // 3개 모델 모두 실행 → 탐지된 위반 유형 후보 수집
        val yoloResults = LABELS.associateWith { label -> runYolo(originalFrame, label) }
        val yoloDetected = yoloResults.filter { it.value.isNotEmpty() }

        // YOLO가 아무것도 감지 못하면 → 정상 주행으로 판단, LRCN 건너뜀
        if (yoloDetected.isEmpty()) {
            consecutiveCount = 0
            lastDetectedType = ""
            Log.d("ViolationDetector", "YOLO 감지 없음 → 정상 주행")
            onDebugUpdate?.invoke(DetectionDebugState(
                isMoving = isMoving, yoloBoxes = emptyMap(), yoloDetected = false,
                frameW = originalFrame.width, frameH = originalFrame.height
            ))
            return null
        }

        // 가장 높은 confidence를 가진 YOLO 결과를 1차 후보로
        val yoloPrimaryLabel = yoloDetected
            .maxByOrNull { entry -> entry.value.maxOf { it[4] } }!!.key
        Log.d("ViolationDetector", "YOLO 1차 감지: $yoloPrimaryLabel (${yoloDetected.keys})")

        // ── [2단계] 모션 게이트 ───────────────────────────
        if (!isMoving && yoloPrimaryLabel != "신호위반") {
            Log.d("ViolationDetector", "정지 상태 → ${yoloPrimaryLabel} 스킵")
            consecutiveCount = 0
            lastDetectedType = ""
            onDebugUpdate?.invoke(DetectionDebugState(
                isMoving = false,
                yoloBoxes = yoloDetected.mapValues { it.value.map { d -> d } },
                yoloDetected = true,
                frameW = originalFrame.width, frameH = originalFrame.height
            ))
            return null
        }

        // ── [3단계] LRCN으로 유형 확인 ───────────────────
        val frames = frameBuffer.toList()
        val inputBuffer = ByteBuffer.allocateDirect(
            1 * SEQUENCE_LENGTH * IMAGE_SIZE * IMAGE_SIZE * 3 * 4
        ).apply { order(ByteOrder.nativeOrder()) }

        frames.forEach { frame ->
            val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
            frame.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
            pixels.forEach { pixel ->
                inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
                inputBuffer.putFloat(((pixel shr 8)  and 0xFF) / 255.0f)
                inputBuffer.putFloat((pixel           and 0xFF) / 255.0f)
            }
        }
        inputBuffer.rewind()

        val output = Array(1) { FloatArray(3) }
        lrcnInterpreter?.run(inputBuffer, output)

        val probs      = output[0]
        val maxIdx     = probs.indices.maxByOrNull { probs[it] } ?: return null
        val confidence = probs[maxIdx]
        val lrcnLabel  = LABELS[maxIdx]

        // LRCN confidence가 낮으면 YOLO 1차 후보를 사용, 높으면 LRCN 우선
        val finalLabel = if (confidence >= CONFIDENCE_THRESHOLD) lrcnLabel else yoloPrimaryLabel
        Log.d("ViolationDetector", "LRCN: $lrcnLabel (${"%.0f".format(confidence * 100)}%), 최종: $finalLabel")

        // ── [4단계] 보정 필터 ─────────────────────────────
        // 신호위반: 실제 빨간 신호등 확인
        if (finalLabel == "신호위반" && !hasTrafficLightRed(originalFrame)) {
            Log.d("ViolationDetector", "신호등 미감지 → 신호위반 스킵")
            consecutiveCount = 0
            lastDetectedType = ""
            return null
        }

        // HSV 보정 (진로변경 → 중앙선침범 교정)
        val correctedLabel = hsvCorrect(finalLabel, originalFrame)

        if (crossedStopLine && correctedLabel == "신호위반") {
            Log.d("ViolationDetector", "정지선 통과 + 신호위반 일치 → 확정")
        }

        // ── [5단계] 연속 감지 카운트 ─────────────────────
        if (correctedLabel == lastDetectedType) {
            consecutiveCount++
        } else {
            consecutiveCount = 1
            lastDetectedType = correctedLabel
        }

        // YOLO + LRCN 모두 같은 유형이면 연속 기준 1 낮춤
        val yoloLrcnAgree = yoloDetected.containsKey(correctedLabel) && lrcnLabel == correctedLabel
        val required = (CONFIRM_COUNTS[correctedLabel] ?: 2).let {
            if (yoloLrcnAgree) (it - 1).coerceAtLeast(1) else it
        }
        val isConfirmed = consecutiveCount >= required
        onDebugUpdate?.invoke(DetectionDebugState(
            isMoving = isMoving,
            yoloBoxes = yoloDetected.mapValues { it.value.map { d -> d } },
            yoloDetected = true,
            lrcnLabel = lrcnLabel,
            lrcnConf = confidence,
            finalLabel = correctedLabel,
            consecutive = consecutiveCount,
            required = required,
            confirmed = isConfirmed,
            frameW = originalFrame.width,
            frameH = originalFrame.height
        ))

        if (!isConfirmed) return null

        consecutiveCount = 0
        lastDetectedType = ""

        return DetectionResult(
            violationType = correctedLabel,
            confidence    = confidence,
            probabilities = probs
        )
    }

    // ── 정상 주행 게이트: 프레임 차분 ────────────────────────
    // 연속 두 프레임의 그레이스케일 평균 차분 < MOTION_THRESHOLD → 정지
    private fun isVehicleMoving(bitmap: Bitmap): Boolean {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        // 그레이스케일 변환 (10픽셀마다 샘플링 — 속도 최적화)
        val step = 10
        val gray = IntArray(pixels.size / step + 1)
        var idx  = 0
        for (i in pixels.indices step step) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8)  and 0xFF
            val b = p           and 0xFF
            gray[idx++] = (r * 299 + g * 587 + b * 114) / 1000
        }

        val prev = prevGray
        prevGray = gray

        if (prev == null || prev.size != gray.size) return true

        var diff = 0L
        for (i in gray.indices) diff += Math.abs(gray[i] - prev[i])
        val avgDiff = diff / gray.size

        return avgDiff >= MOTION_THRESHOLD
    }

    // ── 브레이크등 필터: 화면 상단 빨간 원형 영역 확인 ──────
    // 신호등은 화면 상단 35% 이내에 고정된 빨간 원
    // 브레이크등은 하단 + 움직이는 차량에 붙어있음
    private fun hasTrafficLightRed(bitmap: Bitmap): Boolean {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val upperLimit = (h * 0.35f).toInt()
        val hsv = FloatArray(3)
        var redCount = 0

        for (j in 0 until upperLimit) {
            for (i in 0 until w) {
                val p = pixels[j * w + i]
                val r = (p shr 16) and 0xFF
                val g = (p shr 8)  and 0xFF
                val b = p           and 0xFF
                android.graphics.Color.RGBToHSV(r, g, b, hsv)
                // 빨간색: Hue 0~10 또는 350~360, 채도·명도 충분
                if (hsv[1] > 0.5f && hsv[2] > 0.4f &&
                    (hsv[0] <= 10f || hsv[0] >= 350f)) {
                    redCount++
                }
            }
        }

        val upperArea = upperLimit * w
        // 상단 영역의 0.1% 이상이 빨간색이면 신호등으로 판단
        return redCount.toFloat() / upperArea > 0.001f
    }

    // ── HSV 보정 고도화: 노란선 방향 확인 ────────────────────
    // 단순 픽셀 비율뿐 아니라 세로 방향 분포 확인
    // 중앙선은 주행 방향과 평행 → 열(column) 방향으로 분포
    private fun hsvCorrect(violationType: String, frame: Bitmap): String {
        if (violationType != "진로변경위반") return violationType

        val width  = frame.width
        val height = frame.height
        val pixels = IntArray(width * height)
        frame.getPixels(pixels, 0, width, 0, 0, width, height)

        val hsv = FloatArray(3)
        var totalYellow = 0
        val colCounts = IntArray(width)   // 열별 노란 픽셀 수
        val rowCounts = IntArray(height)  // 행별 노란 픽셀 수

        // 하단 절반만 분석 (도로 영역)
        val startRow = height / 2
        for (j in startRow until height) {
            for (i in 0 until width) {
                val p = pixels[j * width + i]
                val r = (p shr 16) and 0xFF
                val g = (p shr 8)  and 0xFF
                val b = p           and 0xFF
                android.graphics.Color.RGBToHSV(r, g, b, hsv)
                if (hsv[1] > 0.3f && hsv[2] > 0.3f && hsv[0] in 20f..70f) {
                    colCounts[i]++
                    rowCounts[j]++
                    totalYellow++
                }
            }
        }

        val ratio = totalYellow.toFloat() / (width * (height - startRow))
        if (ratio < 0.005f) return violationType

        // 세로 방향(열) 집중도 vs 가로 방향(행) 집중도 비교
        // 중앙선은 세로선 → 특정 열에 픽셀이 집중됨
        val maxCol = colCounts.maxOrNull() ?: 0
        val maxRow = rowCounts.maxOrNull() ?: 0

        return if (maxCol >= maxRow) {
            // 세로 방향 집중 → 중앙선
            Log.d("ViolationDetector", "HSV 보정: 세로 노란선 감지 → 중앙선침범")
            "중앙선침범"
        } else {
            // 가로 방향 집중 → 정지선 색 or 노란 차량 → 보정 안 함
            violationType
        }
    }

    // ── 정지선 감지: 화면 하단 수평 흰선 ────────────────────
    // 화면 하단 40% ROI에서 흰색 픽셀이 가로 방향으로 STOP_LINE_RATIO 이상인 행 존재
    private fun detectStopLine(bitmap: Bitmap): Boolean {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val startRow = (h * 0.6f).toInt()

        for (j in startRow until h) {
            var whiteCount = 0
            for (i in 0 until w) {
                val p = pixels[j * w + i]
                val r = (p shr 16) and 0xFF
                val g = (p shr 8)  and 0xFF
                val b = p           and 0xFF
                val brightness  = (r + g + b) / 3
                val maxC = maxOf(r, g, b)
                val minC = minOf(r, g, b)
                val saturation = if (maxC > 0) (maxC - minC).toFloat() / maxC else 0f
                // 흰색: 높은 밝기 + 낮은 채도
                if (brightness > 200 && saturation < 0.15f) whiteCount++
            }
            if (whiteCount.toFloat() / w >= STOP_LINE_RATIO) return true
        }
        return false
    }

    fun runYolo(bitmap: Bitmap, violationType: String): List<FloatArray> {
        val session = yoloSessions[violationType] ?: return emptyList()

        val resized     = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
        val inputTensor = bitmapToOnnxTensor(resized)

        val results = session.run(mapOf("images" to inputTensor))
        val output  = results[0].value as Array<*>

        return parseYoloOutput(output)
    }

    private fun bitmapToOnnxTensor(bitmap: Bitmap): OnnxTensor {
        val floatArray = FloatArray(1 * 3 * 640 * 640)
        val pixels     = IntArray(640 * 640)
        bitmap.getPixels(pixels, 0, 640, 0, 0, 640, 640)

        pixels.forEachIndexed { i, pixel ->
            floatArray[i]               = ((pixel shr 16) and 0xFF) / 255.0f
            floatArray[i + 640 * 640]   = ((pixel shr 8)  and 0xFF) / 255.0f
            floatArray[i + 640 * 640*2] = (pixel           and 0xFF) / 255.0f
        }

        val floatBuffer = java.nio.FloatBuffer.wrap(floatArray)
        return OnnxTensor.createTensor(ortEnvironment, floatBuffer, longArrayOf(1, 3, 640, 640))
    }

    // YOLOv5 출력 파싱: shape (1, 25200, 5+classes)
    // [cx, cy, w, h, obj_conf, cls1, cls2, ...]
    // obj_conf * max(cls_conf) > 임계값인 박스만 반환
    private fun parseYoloOutput(output: Array<*>): List<FloatArray> {
        val boxes = mutableListOf<FloatArray>()
        // ONNX Runtime → Array<Array<FloatArray>> (batch, anchors, values)
        val batch = output as? Array<*> ?: return boxes
        val anchors = batch[0] as? Array<*> ?: return boxes
        for (anchor in anchors) {
            val det = anchor as? FloatArray ?: continue
            if (det.size < 5) continue
            val objConf = det[4]
            val maxCls  = if (det.size > 5) det.drop(5).maxOrNull() ?: 0f else 1f
            val conf    = objConf * maxCls
            if (conf > 0.4f) boxes.add(det)
        }
        return boxes
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream    = FileInputStream(fileDescriptor.fileDescriptor)
        return inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    fun getBufferStatus(): Pair<Int, Int> = Pair(frameBuffer.size, SEQUENCE_LENGTH)

    fun release() {
        frameBuffer.forEach { it.recycle() }
        frameBuffer.clear()
        lrcnInterpreter?.close()
        yoloSessions.values.forEach { it.close() }
        ortEnvironment?.close()
    }
}
