package com.traffic.violation

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/**
 * OverlayView.kt
 *
 * 카메라 프리뷰 위에 추론 상태를 실시간 시각화
 *   - YOLO 감지 박스
 *   - Motion / YOLO gate / LRCN / 최종 판정 패널
 *   - 위반 확정 시 하단 배너
 */

data class DetectionDebugState(
    val isMoving: Boolean = true,
    val yoloBoxes: Map<String, List<FloatArray>> = emptyMap(), // type → [[x1,y1,x2,y2,conf]]
    val yoloDetected: Boolean = false,
    val lrcnLabel: String = "-",
    val lrcnConf: Float = 0f,
    val finalLabel: String = "-",
    val consecutive: Int = 0,
    val required: Int = 0,
    val confirmed: Boolean = false,
    val frameW: Int = 640,
    val frameH: Int = 640
)

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var state = DetectionDebugState()

    // ── 색상 ──────────────────────────────────────────────────
    private val colorMap = mapOf(
        "신호위반"    to Color.rgb(255, 60,  60),
        "중앙선침범"  to Color.rgb(255, 160, 0),
        "진로변경위반" to Color.rgb(60,  220, 60),
        "정상"       to Color.rgb(160, 160, 160),
    )

    // ── Paint ─────────────────────────────────────────────────
    private val boxPaint   = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 4f }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val textPaint  = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textSize = 36f; typeface = Typeface.DEFAULT_BOLD
    }
    private val smallText  = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textSize = 28f
    }
    private val panelPaint = Paint().apply { color = Color.argb(170, 20, 20, 20); style = Paint.Style.FILL }
    private val barBgPaint = Paint().apply { color = Color.argb(180, 60, 60, 60); style = Paint.Style.FILL }
    private val threshPaint = Paint().apply { color = Color.YELLOW; strokeWidth = 3f; style = Paint.Style.STROKE }
    private val bannerPaint = Paint().apply { style = Paint.Style.FILL }

    fun updateState(newState: DetectionDebugState) {
        state = newState
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val vw = width.toFloat()
        val vh = height.toFloat()
        if (vw == 0f || vh == 0f) return

        val scaleX = vw / state.frameW.toFloat()
        val scaleY = vh / state.frameH.toFloat()

        // ── 1. YOLO 박스 ──────────────────────────────────────
        for ((vtype, boxes) in state.yoloBoxes) {
            val color = colorMap[vtype] ?: Color.WHITE
            boxPaint.color = color
            labelPaint.color = Color.argb(180, Color.red(color), Color.green(color), Color.blue(color))
            for (box in boxes) {
                if (box.size < 5) continue
                val x1 = box[0] * scaleX
                val y1 = box[1] * scaleY
                val x2 = box[2] * scaleX
                val y2 = box[3] * scaleY
                val conf = box[4]
                canvas.drawRect(x1, y1, x2, y2, boxPaint)
                val label = "car(${vtype.take(2)}) ${"%.0f".format(conf * 100)}%"
                val labelW = textPaint.measureText(label) + 12f
                canvas.drawRect(x1, y1 - 42f, x1 + labelW, y1, labelPaint)
                canvas.drawText(label, x1 + 6f, y1 - 10f, textPaint)
            }
        }

        // ── 2. 상태 패널 (좌상단) ─────────────────────────────
        val px = 16f
        val py = 16f
        val pw = 380f
        val ph = 220f
        canvas.drawRoundRect(px, py, px + pw, py + ph, 16f, 16f, panelPaint)

        val lineH = 42f
        // Motion
        val motionColor = if (state.isMoving) Color.rgb(0, 220, 100) else Color.rgb(120, 120, 220)
        smallText.color = motionColor
        canvas.drawText(
            "Motion : ${if (state.isMoving) "Moving" else "Stopped"}",
            px + 12f, py + lineH, smallText
        )

        // YOLO gate
        val yoloStr = if (state.yoloDetected) state.yoloBoxes.keys.joinToString(", ") else "None"
        smallText.color = if (state.yoloDetected) Color.rgb(0, 200, 255) else Color.GRAY
        canvas.drawText("YOLO   : $yoloStr", px + 12f, py + lineH * 2, smallText)

        // LRCN
        val lrcnColor = colorMap[state.lrcnLabel] ?: Color.LTGRAY
        smallText.color = lrcnColor
        canvas.drawText(
            "LRCN   : ${state.lrcnLabel} (${"%.0f".format(state.lrcnConf * 100)}%)",
            px + 12f, py + lineH * 3, smallText
        )

        // confidence 바
        val barX = px + 12f
        val barY = py + lineH * 3 + 8f
        val barW = pw - 24f
        val barH = 14f
        canvas.drawRoundRect(barX, barY, barX + barW, barY + barH, 6f, 6f, barBgPaint)
        val fillW = barW * state.lrcnConf.coerceIn(0f, 1f)
        val fillPaint = Paint().apply {
            color = lrcnColor; style = Paint.Style.FILL
        }
        canvas.drawRoundRect(barX, barY, barX + fillW, barY + barH, 6f, 6f, fillPaint)
        // 임계선 (0.7)
        val threshX = barX + barW * 0.7f
        canvas.drawLine(threshX, barY - 4f, threshX, barY + barH + 4f, threshPaint)

        // 최종 판정
        val finalColor = colorMap[state.finalLabel] ?: Color.LTGRAY
        smallText.color = finalColor
        canvas.drawText(
            "Final  : ${state.finalLabel}  (${state.consecutive}/${state.required})",
            px + 12f, py + lineH * 4 + barH + 4f, smallText
        )

        // ── 3. 위반 확정 배너 (하단) ─────────────────────────
        if (state.confirmed) {
            val bannerH = 90f
            val bc = colorMap[state.finalLabel] ?: Color.RED
            bannerPaint.color = Color.argb(220, Color.red(bc), Color.green(bc), Color.blue(bc))
            canvas.drawRect(0f, vh - bannerH, vw, vh, bannerPaint)
            textPaint.textSize = 48f
            textPaint.color = Color.WHITE
            val msg = "!! ${state.finalLabel} 위반 확정 !!"
            val tw = textPaint.measureText(msg)
            canvas.drawText(msg, (vw - tw) / 2f, vh - 24f, textPaint)
            textPaint.textSize = 36f
        }
    }
}
