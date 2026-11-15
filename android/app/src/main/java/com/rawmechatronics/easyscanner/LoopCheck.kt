package com.rawmechatronics.easyscanner

import android.content.Context
import android.util.Log
import java.io.File
import org.opencv.android.OpenCVLoader
import org.opencv.calib3d.Calib3d
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * In-app relocalisation check: runs SuperPoint on the first and last keyframe of
 * a scan, matches their descriptors and reports how strongly the two views of
 * the same scene connect. This is the front-end the loop-closure stage uses.
 */
object LoopCheck {
    private const val TAG = "SpLoop"
    private const val W = 640
    private const val H = 480

    private class Feat(val kp: FloatArray, val desc: FloatArray, val n: Int)

    private fun inliers(p1: MutableList<Point>, p2: MutableList<Point>, k1: DoubleArray, k2: DoubleArray): Int {
        if (p1.size < 8) return -1
        if (!OpenCVLoader.initLocal()) return -2
        val a = MatOfPoint2f(*p1.toTypedArray())
        val b = MatOfPoint2f(*p2.toTypedArray())
        val k1m = Mat(3, 3, org.opencv.core.CvType.CV_64F)
        val k2m = Mat(3, 3, org.opencv.core.CvType.CV_64F)
        k1m.put(0, 0, *doubleArrayOf(k1[0], 0.0, k1[2], 0.0, k1[1], k1[3], 0.0, 0.0, 1.0))
        k2m.put(0, 0, *doubleArrayOf(k2[0], 0.0, k2[2], 0.0, k2[1], k2[3], 0.0, 0.0, 1.0))
        val e = Calib3d.findEssentialMat(a, b, k1m, Calib3d.RANSAC, 0.999, 1.5)
        val E = DoubleArray(9)
        e.get(0, 0, E)
        var n = 0
        val t = 2.0 / k1[0]
        for (i in 0 until p1.size) {
            val x1 = (p1[i].x - k1[2]) / k1[0]
            val y1 = (p1[i].y - k1[3]) / k1[1]
            val x2 = (p2[i].x - k2[2]) / k2[0]
            val y2 = (p2[i].y - k2[3]) / k2[1]
            val ex = x2 * (E[0] * x1 + E[1] * y1 + E[2]) + y2 * (E[3] * x1 + E[4] * y1 + E[5]) +
                (E[6] * x1 + E[7] * y1 + E[8])
            val a1 = E[0] * x2 + E[3] * y2 + E[6]
            val a2 = E[1] * x2 + E[4] * y2 + E[7]
            val b1 = E[0] * x1 + E[1] * y1 + E[2]
            val b2 = E[3] * x1 + E[4] * y1 + E[5]
            val d = ex * ex / (a1 * a1 + a2 * a2 + b1 * b1 + b2 * b2 + 1e-12)
            if (d < t * t) n++
        }
        return n
    }

    private fun intrinsicsFor(dir: File, name: String): DoubleArray? {
        val f = File(dir, "keyframes.jsonl")
        if (!f.exists()) return null
        for (line in f.readLines()) {
            if (line.isBlank()) continue
            val o = org.json.JSONObject(line)
            if (o.optString("file") == name) {
                val k = o.getJSONArray("k")
                return doubleArrayOf(k.getDouble(0), k.getDouble(1), k.getDouble(2), k.getDouble(3))
            }
        }
        return null
    }

    fun relocalisation(ctx: Context, first: File, last: File): String {
        val a = features(ctx, first) ?: return "first inference failed"
        val b = features(ctx, last) ?: return "last inference failed"
        if (a.n < 50 || b.n < 50) return "too few keypoints: " + a.n + "/" + b.n
        var matches = 0
        val mp1 = ArrayList<Point>()
        val mp2 = ArrayList<Point>()
        var ratioSum = 0.0
        for (i in 0 until a.n) {
            var best = -1
            var bestS = -2f
            var secondS = -2f
            for (j in 0 until b.n) {
                var s = 0f
                val oa = i * 256
                val ob = j * 256
                for (k in 0 until 256) s += a.desc[oa + k] * b.desc[ob + k]
                if (s > bestS) {
                    secondS = bestS
                    bestS = s
                    best = j
                } else if (s > secondS) {
                    secondS = s
                }
            }
            if (best >= 0 && bestS > 0.6f && secondS < bestS - 0.05f) {
                matches++
                ratioSum += (bestS - secondS).toDouble()
                mp1.add(Point(a.kp[i * 2].toDouble(), a.kp[i * 2 + 1].toDouble()))
                mp2.add(Point(b.kp[best * 2].toDouble(), b.kp[best * 2 + 1].toDouble()))
            }
        }
        val purity = if (matches > 0) ratioSum / matches else 0.0
        val k1 = intrinsicsFor(first.parentFile, first.name)
        val k2 = intrinsicsFor(last.parentFile, last.name)
        var inl = -3
        if (k1 != null && k2 != null) {
            inl = try { inliers(mp1, mp2, k1, k2) } catch (ex: Exception) { -4 }
        }
        return "kp=" + a.n + "/" + b.n + " matches=" + matches +
            " mean_margin=" + String.format("%.3f", purity) + " essential_inliers=" + inl
    }

    private fun features(ctx: Context, jpeg: File): Feat? {
        val log = SuperPointRunner.runOnJpeg(ctx, jpeg)
        Log.i(TAG, "infer " + jpeg.name + ": " + log)
        if (!log.contains("rc=0")) return null
        val out = File(ctx.filesDir, "sp/out")
        val scores = find(out, "scores") ?: return null
        val descFile = find(out, "descriptors") ?: return null
        val scoresF = readHalf(scores)
        if (scoresF.size < W * H) return null
        val kpList = ArrayList<Float>()
        for (y in 4 until H - 4) {
            for (x in 4 until W - 4) {
                val v = scoresF[y * W + x]
                if (v <= 0.001f) continue
                var isMax = true
                var dy = -4
                while (dy <= 4 && isMax) {
                    var dx = -4
                    while (dx <= 4 && isMax) {
                        if ((dx != 0 || dy != 0) && scoresF[(y + dy) * W + (x + dx)] > v) isMax = false
                        dx++
                    }
                    dy++
                }
                if (isMax) {
                    kpList.add(x.toFloat())
                    kpList.add(y.toFloat())
                    if (kpList.size >= 2000) break
                }
            }
            if (kpList.size >= 2000) break
        }
        val n = kpList.size / 2
        if (n == 0) return null
        val desc = readHalf(descFile)
        val cells = 60 * 80
        if (desc.size < 256 * cells) return null
        val outDesc = FloatArray(n * 256)
        for (i in 0 until n) {
            val cx = (kpList[i * 2].toInt() / 8).coerceIn(0, 79)
            val cy = (kpList[i * 2 + 1].toInt() / 8).coerceIn(0, 59)
            for (c in 0 until 256) {
                outDesc[i * 256 + c] = desc[c * cells + cy * 80 + cx]
            }
        }
        return Feat(kpList.toFloatArray(), outDesc, n)
    }

    private fun find(dir: File, needle: String): File? {
        val files = dir.listFiles() ?: return null
        for (f in files) {
            if (f.isDirectory) find(f, needle)?.let { return it }
            else if (f.name.contains(needle) && f.name.endsWith(".raw")) return f
        }
        return null
    }

    private fun readHalf(file: File): FloatArray {
        val bytes = file.readBytes()
        val n = bytes.size / 2
        val out = FloatArray(n)
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until n) out[i] = halfToFloat(bb.short)
        return out
    }

    private fun halfToFloat(h: Short): Float {
        val bits = h.toInt() and 0xffff
        val sign = (bits and 0x8000) shl 16
        val exp = (bits ushr 10) and 0x1f
        val mant = bits and 0x3ff
        return when {
            exp == 0 -> java.lang.Float.intBitsToFloat(sign or (mant shl 13))
            exp == 0x1f -> java.lang.Float.intBitsToFloat(sign or 0x7f800000 or (mant shl 13))
            else -> java.lang.Float.intBitsToFloat(sign or ((exp - 15 + 127) shl 23) or (mant shl 13))
        }
    }
}
