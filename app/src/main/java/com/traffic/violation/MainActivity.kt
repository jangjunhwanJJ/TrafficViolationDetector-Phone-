package com.traffic.violation

/**
 * MainActivity.kt
 *
 * 역할: 앱 메인 화면
 *
 * 흐름:
 *   1. 카메라 권한 요청
 *   2. CameraX로 카메라 시작
 *   3. 매 프레임을 ViolationDetector + VideoRecorder에 전달
 *   4. 위반 감지 시 UI 업데이트 + 저장 트리거
 */

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.traffic.violation.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var detector: ViolationDetector
    private lateinit var recorder: VideoRecorder
    private lateinit var cameraExecutor: ExecutorService

    private var isDetecting = false
    private var isCameraStarted = false

    companion object {
        private const val REQUEST_CAMERA_PERMISSION = 100
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // AI 모델 + 녹화기 초기화
        detector = ViolationDetector(this)
        recorder = VideoRecorder(this)

        // 백그라운드 스레드 (카메라 이미지 분석용)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // 저장 완료 시 토스트 표시
        recorder.onSaveComplete = { filePath, violationType ->
            runOnUiThread {
                Toast.makeText(
                    this,
                    "저장 완료: $violationType\n$filePath",
                    Toast.LENGTH_LONG
                ).show()
                binding.tvRecording.visibility = View.GONE
            }
        }

        // 감지 시작/일시정지/재개 버튼
        binding.btnToggle.isEnabled = false  // 모델 로드 완료 전까지 비활성
        binding.btnToggle.setOnClickListener {
            when {
                !isDetecting && !isCameraStarted -> startDetection()   // 최초 시작
                !isDetecting -> resumeDetection()                       // 일시정지 후 재개
                else -> pauseDetection()                                // 감지 중 → 일시정지
            }
        }

        // 카메라 권한 확인
        if (hasCameraPermission()) {
            initModels()
        } else {
            requestCameraPermission()
        }
    }

    // 최초 시작: 카메라 ON + 감지 시작
    private fun startDetection() {
        isDetecting = true
        isCameraStarted = true
        binding.btnToggle.text = "일시정지"
        binding.tvStatus.text = "감지 중..."
        startCamera()
    }

    // 일시정지 후 재개 (카메라는 이미 켜져 있음)
    private fun resumeDetection() {
        isDetecting = true
        binding.btnToggle.text = "일시정지"
        binding.tvStatus.text = "감지 중..."
    }

    // 카메라는 유지, 추론·녹화만 중단
    private fun pauseDetection() {
        isDetecting = false
        binding.btnToggle.text = "재개"
        binding.tvStatus.text = "일시정지됨"
    }

    private fun initModels() {
        // 모델 로드는 시간이 걸리므로 백그라운드에서 실행
        cameraExecutor.execute {
            try {
                detector.initialize()
                runOnUiThread {
                    // 로드 완료 → 버튼 활성화 (사용자가 직접 시작)
                    binding.tvStatus.text = "준비 완료 — 감지 시작을 누르세요"
                    binding.btnToggle.text = "감지 시작"
                    binding.btnToggle.isEnabled = true
                }
            } catch (e: Exception) {
                runOnUiThread {
                    binding.tvStatus.text = "모델 로드 실패: ${e.message}"
                    Log.e("MainActivity", "모델 로드 실패", e)
                }
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // 카메라 프리뷰
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            // 이미지 분석 (매 프레임 처리)
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e("MainActivity", "카메라 시작 실패", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        if (!isDetecting) {
            imageProxy.close()
            return
        }

        // ImageProxy → Bitmap 변환
        val bitmap = imageProxy.toBitmap()
        imageProxy.close()

        // 녹화 버퍼에 프레임 추가
        recorder.addFrame(bitmap)

        // 위반 감지 추론
        val result = detector.processFrame(bitmap)

        // UI 업데이트 (메인 스레드)
        runOnUiThread {
            // 버퍼 상태 표시
            val (current, total) = detector.getBufferStatus()
            binding.tvBuffer.text = "버퍼: $current/$total"

            // 녹화 중 표시
            binding.tvRecording.visibility =
                if (recorder.isSavingNow()) View.VISIBLE else View.GONE

            // 위반 감지 결과 표시
            if (result != null) {
                val percent = (result.confidence * 100).toInt()
                binding.tvStatus.text =
                    "⚠️ ${result.violationType} 감지! ($percent%)"
                binding.tvStatus.setBackgroundColor(0xAAFF0000.toInt())

                // 저장 트리거
                if (!recorder.isSavingNow()) {
                    recorder.triggerSave(result.violationType)
                    binding.tvRecording.visibility = View.VISIBLE
                }
            } else {
                binding.tvStatus.text = "감지 중..."
                binding.tvStatus.setBackgroundColor(0xAA000000.toInt())
            }
        }
    }

    // 권한 관련
    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            REQUEST_CAMERA_PERMISSION
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA_PERMISSION &&
            grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            initModels()
        } else {
            Toast.makeText(this, "카메라 권한이 필요합니다", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.release()
        recorder.release()
        cameraExecutor.shutdown()
    }
}
