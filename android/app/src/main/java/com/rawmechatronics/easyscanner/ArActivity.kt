package com.rawmechatronics.easyscanner

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.HitResult
import com.google.ar.core.Plane
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * ARCore-pose prototype: same laser engine as MainActivity, but the camera pose
 * comes from ARCore instead of the ChArUco board. No board is needed during a
 * scan; only the saved laser-plane calibration is used.
 */
class ArActivity : Activity(), GLSurfaceView.Renderer, FrameProcessor.Listener {

    companion object {
        private const val TAG = "EasyScannerAr"
        private const val REQUEST_CAMERA = 7
        /** Synthetic scan volume (mm) used when there is no board to measure it. */
        private const val VOLUME_MM = 400.0
        private val AR_CONFIG = """
            {"camera":{"id":"0"},
             "board":{"squares_x":8,"squares_y":8,"square_mm":50.0,"marker_mm":37.0,"dict":5},
             "volume":{"height_mm":400.0,"margin_mm":20.0},
             "draw":{"grid":false,"axes":false,"corners":false,"volume":true,"cloud":true},
             "laser":{"enabled":true,"color":"red","thresh":12,"min_sat":40,"without_board":true},
             "calib_views":64,"plane_views":10,"max_cloud_points":6000000}
        """.trimIndent()
    }

    private lateinit var surface: GLSurfaceView
    private lateinit var status: TextView
    private var session: Session? = null
    private var processor: FrameProcessor? = null
    private var installRequested = false
    private var resumed = false

    /** Anchor point on a detected plane: the origin of the scan frame. */
    private var anchor: FloatArray? = null
    private var scanOn = false
    private var pointCount = 0

    // GL
    private var program = 0
    private var textureId = 0
    private var uTex = 0
    private var quadCoords: FloatBuffer = FloatBuffer.allocate(8)
    private var quadTexCoords: FloatBuffer = FloatBuffer.allocate(8)
    private var viewTx = 0f
    private var viewTy = 0f
    private var tapPending: FloatArray? = null
    private var frames = 0
    private var loggedTex = false
    private val io = Executors.newSingleThreadExecutor()
    private var kfDir: File? = null
    private var kfIndex = 0
    private var lastKfMs = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        surface = GLSurfaceView(this).apply {
            preserveEGLContextOnPause = true
            setEGLContextClientVersion(2)
            setRenderer(this@ArActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }
        surface.setOnTouchListener { _, e -> onSurfaceTouch(e) }
        status = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt())
            setBackgroundColor(0xAA000000.toInt())
            textSize = 13f
            setPadding(24, 24, 24, 24)
            text = "Looking for ARCore…"
        }
        val root = FrameLayout(this)
        root.addView(surface, FrameLayout.LayoutParams(-1, -1))
        root.addView(status, FrameLayout.LayoutParams(-1, -2, Gravity.TOP))
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setBackgroundColor(0xCC000000.toInt())
        }
        fun button(label: String, action: () -> Unit) = Button(this).apply {
            text = label
            setOnClickListener { action() }
            bar.addView(this)
        }
        button("Clear origin") { setAnchor(null) }
        button("Scan") { toggleScan() }
        button("Save PLY") { exportPly() }
        root.addView(bar, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))
        setContentView(root)
    }

    // ── lifecycle ──────────────────────────────────────────────────────────

    override fun onResume() {
        super.onResume()
        surface.onResume()
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA)
            return
        }
        try {
            when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                    installRequested = true
                    return
                }
                ArCoreApk.InstallStatus.INSTALLED -> Unit
                null -> Unit
            }
            if (session == null) session = Session(this).also { configureSession(it) }
            session?.resume()
            if (processor == null) {
                processor = FrameProcessor(filesDir.absolutePath, this)
                processor?.configure(AR_CONFIG)
            }
            processor?.command(Cmd.AR_MODE, "1")
            resumed = true
            status.text = "Move the phone. Tap the table next to the object to set the scan origin."
        } catch (e: UnavailableArcoreNotInstalledException) {
            status.text = "ARCore is not installed (Google Play Services for AR)."
        } catch (e: UnavailableApkTooOldException) {
            status.text = "ARCore needs an update."
        } catch (e: UnavailableSdkTooOldException) {
            status.text = "ARCore SDK too old for this device."
        } catch (e: UnavailableDeviceNotCompatibleException) {
            status.text = "This device is not ARCore compatible."
        } catch (e: Exception) {
            Log.e(TAG, "AR session failed", e)
            status.text = "AR session failed: ${e.message}"
        }
    }

    override fun onPause() {
        super.onPause()
        resumed = false
        if (scanOn) {
            scanOn = false
            processor?.command(Cmd.SCAN_STOP)
        }
        session?.pause()
        surface.onPause()
        processor?.release()
        processor = null
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) onResume()
    }

    private fun configureSession(s: Session) {
        val config = Config(s)
        config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
        config.focusMode = Config.FocusMode.AUTO
        config.planeFindingMode = Config.PlaneFindingMode.HORIZONTAL
        config.depthMode = Config.DepthMode.DISABLED
        config.lightEstimationMode = Config.LightEstimationMode.DISABLED
        config.instantPlacementMode = Config.InstantPlacementMode.DISABLED
        s.configure(config)
        val filter = CameraConfigFilter(s).apply {
            setTargetFps(java.util.EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30))
            setDepthSensorUsage(java.util.EnumSet.of(CameraConfig.DepthSensorUsage.DO_NOT_USE))
            setFacingDirection(CameraConfig.FacingDirection.BACK)
        }
        val best = s.getSupportedCameraConfigs(filter)
            .maxByOrNull { it.imageSize.width * it.imageSize.height }
        if (best != null) s.cameraConfig = best
        Log.i(TAG, "camera config: ${best?.imageSize} intrinsics ok")
    }

    // ── UI ─────────────────────────────────────────────────────────────────

    private fun onSurfaceTouch(e: MotionEvent): Boolean {
        if (e.action == MotionEvent.ACTION_UP) tapPending = floatArrayOf(e.x, e.y)
        return true
    }

    private fun setAnchor(p: FloatArray?) {
        anchor = p
        if (p != null) {
            processor?.command(Cmd.CLOUD_CLEAR)
            pointCount = 0
            status.text = "Origin set. Point the laser and press Scan."
        }
    }

    private fun toggleScan() {
        if (anchor == null) {
            Toast.makeText(this, "Tap the table to set the origin first", Toast.LENGTH_SHORT).show()
            return
        }
        if (!scanOn) {
            kfDir = File(getExternalFilesDir(null), "kf-" + System.currentTimeMillis()).apply { mkdirs() }
            kfIndex = 0
            lastKfMs = 0L
            processor?.command(Cmd.CLOUD_CLEAR)
            processor?.command(Cmd.SCAN_START) { msg -> runOnUiThread { status.text = msg } }
            scanOn = true
        } else {
            scanOn = false
            processor?.command(Cmd.SCAN_STOP) { msg -> runOnUiThread { status.text = msg } }
            Log.i(TAG, "keyframes: " + kfIndex + " in " + kfDir)
        }
    }

    private fun exportPly() {
        val dir = getExternalFilesDir(null) ?: filesDir
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val file = File(dir, "ar-scan-$stamp.ply")
        processor?.command(Cmd.EXPORT_PLY, file.absolutePath) { msg ->
            runOnUiThread { status.text = msg }
        }
    }

    // ── FrameProcessor.Listener ────────────────────────────────────────────

    override fun onResult(r: FrameResult) {
        pointCount = r[FrameResult.H.CLOUD_POINTS].toInt()
        if (!scanOn && pointCount > 0) {
            // command results already carry a message; keep the counter visible
        }
        runOnUiThread {
            if (scanOn) status.text = "$pointCount points · scanning"
        }
    }

    override fun onMessages(json: String) {
        Log.i(TAG, "messages: $json")
    }

    override fun onPhotoDue() = Unit

    override fun onError(msg: String) {
        Log.e(TAG, msg)
        runOnUiThread { status.text = msg }
    }

    // ── GL renderer ────────────────────────────────────────────────────────

    override fun onSurfaceCreated(unused: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        Log.i(TAG, "GL vendor=" + GLES20.glGetString(GLES20.GL_VENDOR) + " renderer=" + GLES20.glGetString(GLES20.GL_RENDERER))
        program = buildProgram()
        uTex = GLES20.glGetUniformLocation(program, "uTex")
        val ids = IntArray(1)
        GLES20.glGenTextures(1, ids, 0)
        textureId = ids[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        session?.setCameraTextureName(textureId)
        quadCoords = directFloat(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
        quadTexCoords = directFloat(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f))
    }

    override fun onSurfaceChanged(unused: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewTx = width.toFloat()
        viewTy = height.toFloat()
        try {
            @Suppress("DEPRECATION")
            val rot = windowManager.defaultDisplay.rotation
            session?.setDisplayGeometry(rot, width, height)
        } catch (ex: Exception) {
            Log.w(TAG, "setDisplayGeometry failed: " + ex.message)
        }
    }

    override fun onDrawFrame(unused: GL10?) {
        if (!resumed) return
        val s = session
        if (s == null) {
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
            return
        }
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        s.setCameraTextureName(textureId)
        val frame = try {
            s.update()
        } catch (ex: Exception) {
            Log.w(TAG, "update failed: " + ex.message)
            return
        }
        drawBackground(frame)
        // GL liveness marker: magenta square, bottom-left
        GLES20.glEnable(GLES20.GL_SCISSOR_TEST)
        GLES20.glScissor(0, 0, 48, 48)
        GLES20.glClearColor(1f, 0f, 1f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        GLES20.glDisable(GLES20.GL_SCISSOR_TEST)
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        frames++
        if (frames % 30 == 0) {
            val px = readPixel(viewTx.toInt() / 2, viewTy.toInt() / 2)
            val mk = readPixel(24, 24)
            Log.i(TAG, "diag " + frames + " " + cameraStats(frame) + " center=" + px + " marker=" + mk)
        }
        val state = frame.camera.trackingState
        tapPending?.let { tap ->
            tapPending = null
            if (state == TrackingState.TRACKING) placeAnchor(frame, tap[0], tap[1])
        }
        if (state == TrackingState.TRACKING && anchor != null) processFrame(frame)
        runOnUiThread {
            val cam = cameraStats(frame)
            status.text = "track=" + state + "  cam=" + cam + "  points=" + pointCount
        }
    }

    private fun drawBackground(frame: Frame) {
        GLES20.glUseProgram(program)
        quadCoords.position(0)
        frame.transformCoordinates2d(
            Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, quadCoords,
            Coordinates2d.TEXTURE_NORMALIZED, quadTexCoords,
        )
        quadCoords.position(0)
        quadTexCoords.position(0)
        if (!loggedTex) {
            val arr = FloatArray(8)
            quadTexCoords.get(arr)
            quadTexCoords.position(0)
            Log.i(TAG, "texcoords " + arr.joinToString(","))
            loggedTex = true
        }
        val aPos = GLES20.glGetAttribLocation(program, "aPos")
        val aTex = GLES20.glGetAttribLocation(program, "aTex")
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 0, quadCoords)
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aTex, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)
        GLES20.glEnableVertexAttribArray(aTex)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glUniform1i(uTex, 0)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(aPos)
        GLES20.glDisableVertexAttribArray(aTex)
    }

    /** RGBA of the framebuffer at (x,y) in GL coordinates (origin bottom-left). */
    private fun readPixel(x: Int, y: Int): String {
        val b = java.nio.ByteBuffer.allocateDirect(4)
        GLES20.glReadPixels(x, y, 1, 1, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, b)
        return String.format("%d,%d,%d", b.get(0).toInt() and 0xff, b.get(1).toInt() and 0xff, b.get(2).toInt() and 0xff)
    }

    /** Diagnostics: camera image size and mean luma (sampled Y plane), or why it failed. */
    private fun cameraStats(frame: Frame): String {
        val image = try {
            frame.acquireCameraImage()
        } catch (ex: Exception) {
            return "no image (" + (ex.message ?: "error") + ")"
        }
        try {
            val y = image.planes[0]
            val buf = y.buffer.duplicate()
            buf.rewind()
            val w = image.width
            val h = image.height
            var sum = 0L
            var n = 0
            val row = ByteArray(w)
            var yPos = 0
            var lines = 0
            while (lines < h && buf.remaining() >= w) {
                buf.position(yPos)
                buf.get(row, 0, w)
                for (v in row) { sum += (v.toInt() and 0xff); n++ }
                yPos += y.rowStride
                lines += 4
            }
            val mean = if (n > 0) sum.toDouble() / n else -1.0
            return w.toString() + "x" + h + " luma=" + String.format("%.0f", mean)
        } catch (ex: Exception) {
            return "stats failed"
        } finally {
            image.close()
        }
    }

    private fun placeAnchor(frame: Frame, x: Float, y: Float) {
        val hits = try {
            frame.hitTest(x, y)
        } catch (e: Exception) {
            return
        }
        val hit: HitResult? = hits.firstOrNull { h ->
            h.trackable is Plane && (h.trackable as Plane).trackingState == TrackingState.TRACKING
        }
        val t = hit?.hitPose?.translation
        if (t == null) {
            runOnUiThread { status.text = "No plane under the tap: point at the table around the object." }
            return
        }
        setAnchor(floatArrayOf(t[0], t[1], t[2]))
    }

    private fun processFrame(frame: Frame) {
        val processor = processor ?: return
        val a = anchor ?: return
        val image = try {
            frame.acquireCameraImage()
        } catch (e: Exception) {
            return
        }
        try {
            val planes = image.planes
            if (planes.size < 3) return
            val y = copyDirect(planes[0].buffer)
            val u = copyDirect(planes[1].buffer)
            val v = copyDirect(planes[2].buffer)
            val k = intrinsics(frame, image.width, image.height)
            val pose = boardToCamera(frame, a)
            val now = System.currentTimeMillis()
            if (scanOn && now - lastKfMs > 500) {
                lastKfMs = now
                val dir = kfDir
                if (dir != null) {
                    val planesCopy = arrayOf(
                        copyDirect(planes[0].buffer), copyDirect(planes[1].buffer), copyDirect(planes[2].buffer),
                    )
                    val strides = intArrayOf(planes[0].rowStride, planes[1].rowStride, planes[1].pixelStride)
                    val w = image.width
                    val h = image.height
                    val idx = kfIndex++
                    val poseCopy = pose.copyOf()
                    val kCopy = k.copyOf()
                    val ts = image.timestamp
                    io.execute { saveKeyframe(dir, idx, planesCopy, strides, w, h, ts, poseCopy, kCopy) }
                }
            }
            processor.processArFrame(
                y, planes[0].rowStride,
                u, v, planes[1].rowStride, planes[1].pixelStride,
                image.width, image.height, image.timestamp, pose, k,
            )
        } finally {
            image.close()
        }
    }

    /** ARCore intrinsics re-expressed for the CPU image size, no distortion. */
    private fun intrinsics(frame: Frame, w: Int, h: Int): DoubleArray {
        val i = frame.camera.imageIntrinsics
        val dims = i.getImageDimensions()
        val iw = dims[0]
        val ih = dims[1]
        val sx = if (iw > 0) w.toDouble() / iw else 1.0
        val sy = if (ih > 0) h.toDouble() / ih else 1.0
        val fx = i.focalLength[0].toDouble() * sx
        val fy = i.focalLength[1].toDouble() * sy
        val cx = i.principalPoint[0].toDouble() * sx
        val cy = i.principalPoint[1].toDouble() * sy
        return doubleArrayOf(fx, fy, cx, cy, w.toDouble(), h.toDouble(), 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /**
     * Scan-frame -> camera (OpenCV convention) for this frame. The scan frame is
     * anchored on a horizontal ARCore plane, Z up (gravity), centred on the anchor.
     */
    private fun boardToCamera(frame: Frame, anchor: FloatArray): DoubleArray {
        val m = FloatArray(16)
        frame.camera.pose.toMatrix(m, 0) // column-major, ARCore (GL) camera axes
        // world<-camera with camera axes converted to OpenCV: x same, y and z flipped
        val c0 = doubleArrayOf(m[0].toDouble(), m[1].toDouble(), m[2].toDouble())
        val c1 = doubleArrayOf(-m[4].toDouble(), -m[5].toDouble(), -m[6].toDouble())
        val c2 = doubleArrayOf(-m[8].toDouble(), -m[9].toDouble(), -m[10].toDouble())
        val tw = doubleArrayOf(m[12].toDouble(), m[13].toDouble(), m[14].toDouble())
        // camera<-world = (world<-camera)^T
        val r00 = c0[0]; val r01 = c1[0]; val r02 = c2[0]
        val r10 = c0[1]; val r11 = c1[1]; val r12 = c2[1]
        val r20 = c0[2]; val r21 = c1[2]; val r22 = c2[2]
        val tx = -(r00 * tw[0] + r01 * tw[1] + r02 * tw[2])
        val ty = -(r10 * tw[0] + r11 * tw[1] + r12 * tw[2])
        val tz = -(r20 * tw[0] + r21 * tw[1] + r22 * tw[2])
        // scan frame S in world: x = world X, y = -world Z, z = world Y
        // columns of M (world axes): (1,0,0), (0,0,-1), (0,1,0)
        val half = VOLUME_MM / 2.0
        // t_world_S = anchor - M * [half, half, 0]
        val swx = anchor[0].toDouble() - (half * 1.0 + half * 0.0)
        val swy = anchor[1].toDouble() - (half * 0.0 + half * 1.0)
        val swz = anchor[2].toDouble() - (half * 0.0 + half * -1.0)
        // R_cam_S = R_cw * M ; columns of M as above
        val s0x = r00 * 1.0 + r01 * 0.0 + r02 * 0.0
        val s0y = r10 * 1.0 + r11 * 0.0 + r12 * 0.0
        val s0z = r20 * 1.0 + r21 * 0.0 + r22 * 0.0
        val s1x = r00 * 0.0 + r01 * 0.0 + r02 * -1.0
        val s1y = r10 * 0.0 + r11 * 0.0 + r12 * -1.0
        val s1z = r20 * 0.0 + r21 * 0.0 + r22 * -1.0
        val s2x = r00 * 0.0 + r01 * 1.0 + r02 * 0.0
        val s2y = r10 * 0.0 + r11 * 1.0 + r12 * 0.0
        val s2z = r20 * 0.0 + r21 * 1.0 + r22 * 0.0
        // t_cam_S = t_cam_world + R_cw * t_world_S
        val ex = tx + r00 * swx + r01 * swy + r02 * swz
        val ey = ty + r10 * swx + r11 * swy + r12 * swz
        val ez = tz + r20 * swx + r21 * swy + r22 * swz
        return doubleArrayOf(
            s0x, s1x, s2x, ex,
            s0y, s1y, s2y, ey,
            s0z, s1z, s2z, ez,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /** Convert one ARCore YUV_420_888 frame to a JPEG and append its pose row. */
    private fun saveKeyframe(
        dir: File,
        index: Int,
        planes: Array<ByteBuffer>,
        strides: IntArray,
        w: Int,
        h: Int,
        ts: Long,
        pose: DoubleArray,
        k: DoubleArray,
    ) {
        try {
            val yStride = strides[0]
            val uvStride = strides[1]
            val uvPixel = strides[2]
            val nv21 = ByteArray(w * h + w * h / 2)
            var p = 0
            for (row in 0 until h) {
                planes[0].position(row * yStride)
                planes[0].get(nv21, p, w)
                p += w
            }
            for (row in 0 until h / 2) {
                val vBase = row * uvStride
                for (col in 0 until w / 2) {
                    val vi = vBase + col * uvPixel
                    planes[2].position(vi)
                    nv21[p++] = planes[2].get()
                    planes[1].position(vi)
                    nv21[p++] = planes[1].get()
                }
            }
            val file = File(dir, String.format("kf_%04d.jpg", index))
            FileOutputStream(file).use { out ->
                YuvImage(nv21, android.graphics.ImageFormat.NV21, w, h, null).compressToJpeg(Rect(0, 0, w, h), 90, out)
            }
            val row = org.json.JSONObject().apply {
                put("i", index)
                put("t", ts)
                put("file", file.name)
                put("pose", org.json.JSONArray(pose.toList()))
                put("k", org.json.JSONArray(k.toList()))
            }
            File(dir, "keyframes.jsonl").appendText(row.toString() + System.lineSeparator())
        } catch (ex: Exception) {
            Log.w(TAG, "keyframe " + index + " failed: " + ex.message)
        }
    }

    private fun copyDirect(src: ByteBuffer): ByteBuffer {
        val dup = src.duplicate()
        dup.rewind()
        val out = ByteBuffer.allocateDirect(dup.remaining()).order(ByteOrder.nativeOrder())
        out.put(dup)
        out.rewind()
        return out
    }

    private fun directFloat(values: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
            put(values)
            rewind()
        }

    private fun buildProgram(): Int {
        val vs = """
            attribute vec2 aPos;
            attribute vec2 aTex;
            varying vec2 vTex;
            void main() {
                vTex = aTex;
                gl_Position = vec4(aPos, 0.0, 1.0);
            }
        """.trimIndent()
        val fs = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            varying vec2 vTex;
            uniform samplerExternalOES uTex;
            void main() {
                gl_FragColor = texture2D(uTex, vTex);
            }
        """.trimIndent()
        fun compile(type: Int, src: String): Int {
            val sh = GLES20.glCreateShader(type)
            GLES20.glShaderSource(sh, src)
            GLES20.glCompileShader(sh)
            val ok = IntArray(1)
            GLES20.glGetShaderiv(sh, GLES20.GL_COMPILE_STATUS, ok, 0)
            if (ok[0] == 0) Log.e(TAG, "shader: " + GLES20.glGetShaderInfoLog(sh))
            return sh
        }
        val p = GLES20.glCreateProgram()
        GLES20.glAttachShader(p, compile(GLES20.GL_VERTEX_SHADER, vs))
        GLES20.glAttachShader(p, compile(GLES20.GL_FRAGMENT_SHADER, fs))
        GLES20.glLinkProgram(p)
        return p
    }
}
