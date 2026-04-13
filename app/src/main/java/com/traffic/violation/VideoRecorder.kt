package com.traffic.violation

/**
 * VideoRecorder.kt
 *
 * 역할: 이벤트 트리거 방식 영상 저장
 *
 * 흐름:
 *   1. 원형 버퍼에 항상 최근 5초치 프레임 보관
 *   2. 위반 감지 시 triggerSave() 호출
 *   3. 버퍼(감지 전 5초) + 이후 3초 = 총 8초 영상 저장
 */

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.util.ArrayDeque

class VideoRecorder(private val context: Context) {

    companion object {
        const val FPS = 30
        const val PRE_SECONDS = 5               // 감지 전 보관할 초
        const val POST_SECONDS = 3              // 감지 후 추가 녹화할 초
        const val PRE_BUFFER_SIZE = FPS * PRE_SECONDS    // 150프레임
        const val POST_FRAME_COUNT = FPS * POST_SECONDS  // 90프레임

        const val VIDEO_WIDTH = 480
        const val VIDEO_HEIGHT = 640
        const val BIT_RATE = 2_000_000          // 2Mbps
    }

    // 원형 버퍼: 항상 최근 5초치 원본 프레임 유지
    private val preBuffer = ArrayDeque<Bitmap>(PRE_BUFFER_SIZE)

    private var isSaving = false
    private var isEncoding = false          // 인코딩 시작 후 추가 수집 차단용
    private var postFrameCount = 0
    private val postFrames = mutableListOf<Bitmap>()
    private var currentViolationType = ""
    private var lastSaveTimeMs = 0L
    private val SAVE_COOLDOWN_MS = 15_000L  // 저장 완료 후 15초간 재저장 차단

    // 저장 완료 콜백
    var onSaveComplete: ((filePath: String, violationType: String) -> Unit)? = null

    /**
     * 매 프레임 호출 — 버퍼 유지 및 감지 후 녹화
     */
    @Synchronized
    fun addFrame(bitmap: Bitmap) {
        val copy = bitmap.copy(Bitmap.Config.ARGB_8888, false)

        // 원형 버퍼 유지
        if (preBuffer.size >= PRE_BUFFER_SIZE) {
            preBuffer.pollFirst()?.recycle()
        }
        preBuffer.addLast(copy)

        // isEncoding이 true면 이미 saveVideo() 호출됐으므로 수집 중단
        if (isSaving && !isEncoding) {
            postFrames.add(copy.copy(Bitmap.Config.ARGB_8888, false))
            postFrameCount++

            if (postFrameCount >= POST_FRAME_COUNT) {
                isEncoding = true
                saveVideo()
            }
        }
    }

    /**
     * 위반 감지 시 호출 — 저장 트리거
     * @param violationType 감지된 위반 유형
     */
    @Synchronized
    fun triggerSave(violationType: String) {
        if (isSaving) return
        if (System.currentTimeMillis() - lastSaveTimeMs < SAVE_COOLDOWN_MS) return

        isSaving = true
        isEncoding = false
        postFrameCount = 0
        postFrames.clear()
        currentViolationType = violationType
        Log.d("VideoRecorder", "저장 트리거: $violationType (버퍼 ${preBuffer.size}프레임)")
    }

    /**
     * 실제 영상 저장
     * 감지 전 5초 + 감지 후 3초 = 총 8초
     */
    private fun saveVideo() {
        // 인코딩 스레드와 addFrame() 스레드 간 recycle 충돌 방지
        // → preBuffer 프레임을 독립 복사본으로 스냅샷
        val snapshot = (preBuffer.toList() + postFrames)
            .map { it.copy(Bitmap.Config.ARGB_8888, false) }

        Thread {
            var tempFile: File? = null
            try {
                // 1단계: 캐시 폴더에 인코딩
                tempFile = getTempFile()
                encodeVideo(snapshot, tempFile.absolutePath)

                // 2단계: 인코딩 완료 후 갤러리(MediaStore)로 이동
                val finalPath = moveToGallery(tempFile)
                onSaveComplete?.invoke(finalPath, currentViolationType)
                Log.d("VideoRecorder", "저장 완료: $finalPath")

            } catch (e: Exception) {
                Log.e("VideoRecorder", "저장 실패: ${e.message}")
            } finally {
                tempFile?.delete()
                snapshot.forEach { it.recycle() }
                lastSaveTimeMs = System.currentTimeMillis()  // 쿨다운 시작
                isSaving = false
                isEncoding = false
                postFrameCount = 0
                postFrames.clear()
            }
        }.start()
    }

    /**
     * MediaCodec + MediaMuxer로 H.264 영상 인코딩
     */
    private fun encodeVideo(frames: List<Bitmap>, outputPath: String) {
        val format = MediaFormat.createVideoFormat(
            MediaFormat.MIMETYPE_VIDEO_AVC,
            VIDEO_WIDTH,
            VIDEO_HEIGHT
        ).apply {
            setInteger(MediaFormat.KEY_BIT_RATE, BIT_RATE)
            setInteger(MediaFormat.KEY_FRAME_RATE, FPS)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            setInteger(
                MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
            )
        }

        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()

        val muxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        var trackIndex = -1
        val bufferInfo = MediaCodec.BufferInfo()

        frames.forEachIndexed { frameIdx, bitmap ->
            // 비트맵 → YUV 변환 후 코덱에 입력
            val inputBufferIndex = codec.dequeueInputBuffer(10_000)
            if (inputBufferIndex >= 0) {
                val inputBuffer = codec.getInputBuffer(inputBufferIndex)!!
                val yuvData = bitmapToYuv(bitmap, VIDEO_WIDTH, VIDEO_HEIGHT)
                inputBuffer.put(yuvData)

                val presentationTime = frameIdx * (1_000_000L / FPS)
                val flags = if (frameIdx == frames.size - 1) MediaCodec.BUFFER_FLAG_END_OF_STREAM else 0
                codec.queueInputBuffer(inputBufferIndex, 0, yuvData.size, presentationTime, flags)
            }

            // 출력 버퍼 처리
            var outputDone = false
            while (!outputDone) {
                val outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, 10_000)
                when {
                    outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        trackIndex = muxer.addTrack(codec.outputFormat)
                        muxer.start()
                    }
                    outputBufferIndex >= 0 -> {
                        val outputBuffer = codec.getOutputBuffer(outputBufferIndex)!!
                        if (trackIndex >= 0 && bufferInfo.size > 0) {
                            muxer.writeSampleData(trackIndex, outputBuffer, bufferInfo)
                        }
                        codec.releaseOutputBuffer(outputBufferIndex, false)
                        if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            outputDone = true
                        }
                    }
                    else -> outputDone = true
                }
            }
        }

        codec.stop()
        codec.release()
        muxer.stop()
        muxer.release()
    }

    // Bitmap → YUV420 변환 (MediaCodec 입력 형식)
    private fun bitmapToYuv(bitmap: Bitmap, width: Int, height: Int): ByteArray {
        val scaled = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val pixels = IntArray(width * height)
        scaled.getPixels(pixels, 0, width, 0, 0, width, height)

        val yuv = ByteArray(width * height * 3 / 2)
        var yIndex = 0
        var uvIndex = width * height

        for (j in 0 until height) {
            for (i in 0 until width) {
                val pixel = pixels[j * width + i]
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF

                // RGB → YUV 변환
                val y = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                yuv[yIndex++] = y.coerceIn(0, 255).toByte()

                if (j % 2 == 0 && i % 2 == 0) {
                    val u = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                    val v = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                    yuv[uvIndex++] = u.coerceIn(0, 255).toByte()
                    yuv[uvIndex++] = v.coerceIn(0, 255).toByte()
                }
            }
        }
        return yuv
    }

    // 임시 파일 생성 (인코딩용 — 앱 캐시 폴더)
    private fun getTempFile(): File {
        val dir = File(context.cacheDir, "videos").also { it.mkdirs() }
        return File(dir, "temp_${System.currentTimeMillis()}.mp4")
    }

    // 인코딩 완료된 파일을 갤러리(MediaStore)로 복사
    private fun moveToGallery(tempFile: File): String {
        val timestamp = System.currentTimeMillis()
        val filename = "violation_${currentViolationType}_${timestamp}.mp4"

        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // Android 10 이상
            // IS_PENDING=1 로 먼저 등록 → 갤러리에 즉시 항목 생성
            val values = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, filename)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/TrafficViolation")
                put(MediaStore.Video.Media.IS_PENDING, 1)
            }
            val uri = context.contentResolver.insert(
                MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values
            ) ?: throw Exception("MediaStore insert 실패")

            // 파일 내용 복사
            context.contentResolver.openOutputStream(uri)?.use { output ->
                tempFile.inputStream().use { input -> input.copyTo(output) }
            }

            // IS_PENDING=0 으로 변경 → 갤러리에서 재생 가능 상태
            val updateValues = ContentValues().apply {
                put(MediaStore.Video.Media.IS_PENDING, 0)
            }
            context.contentResolver.update(uri, updateValues, null, null)

            "Movies/TrafficViolation/$filename"
        } else {
            // Android 9 이하: 공개 Movies 폴더에 직접 복사
            val dir = File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES),
                "TrafficViolation"
            ).also { it.mkdirs() }
            val dest = File(dir, filename)
            tempFile.copyTo(dest, overwrite = true)
            dest.absolutePath
        }
    }

    fun isSavingNow(): Boolean = isSaving

    fun release() {
        preBuffer.forEach { it.recycle() }
        preBuffer.clear()
        postFrames.forEach { it.recycle() }
        postFrames.clear()
    }
}
