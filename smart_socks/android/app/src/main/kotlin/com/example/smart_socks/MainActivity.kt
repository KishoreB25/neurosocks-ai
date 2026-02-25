package com.example.smart_socks

import android.app.Activity
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothSocket
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Log
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import java.io.IOException
import java.io.InputStream
import java.util.UUID

class MainActivity : FlutterActivity() {
    companion object {
        private const val TAG = "ClassicBT"
        private const val METHOD_CHANNEL = "com.neurosocks.app/classic_bt"
        private const val DATA_EVENT_CHANNEL = "com.neurosocks.app/classic_bt_data"
        private const val DISCOVERY_EVENT_CHANNEL = "com.neurosocks.app/classic_bt_discovery"
        // Standard SPP UUID for serial port communication
        private val SPP_UUID: UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")
    }

    private var bluetoothAdapter: BluetoothAdapter? = null
    private var bluetoothSocket: BluetoothSocket? = null
    private var inputStream: InputStream? = null
    private var readThread: Thread? = null
    private var isReading = false

    private var dataEventSink: EventChannel.EventSink? = null
    private var discoveryEventSink: EventChannel.EventSink? = null

    private val mainHandler = Handler(Looper.getMainLooper())

    // For Bluetooth enable request
    private var enableBtResult: MethodChannel.Result? = null
    private lateinit var enableBtLauncher: ActivityResultLauncher<Intent>

    // BroadcastReceiver for discovery
    private val discoveryReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                BluetoothDevice.ACTION_FOUND -> {
                    val device: BluetoothDevice? =
                        intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE)
                    device?.let {
                        val info = mapOf(
                            "name" to (it.name ?: ""),
                            "address" to it.address
                        )
                        mainHandler.post { discoveryEventSink?.success(info) }
                        Log.d(TAG, "Found: ${it.name ?: "?"} (${it.address})")
                    }
                }
                BluetoothAdapter.ACTION_DISCOVERY_FINISHED -> {
                    mainHandler.post { discoveryEventSink?.endOfStream() }
                    Log.d(TAG, "Discovery finished")
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: android.os.Bundle?) {
        super.onCreate(savedInstanceState)

        // Register the BT-enable launcher BEFORE configureFlutterEngine
        enableBtLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            val enabled = result.resultCode == Activity.RESULT_OK
            Log.d(TAG, "BT enable result: $enabled")
            mainHandler.post { enableBtResult?.success(enabled) }
            enableBtResult = null
        }
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        val btManager = getSystemService(Context.BLUETOOTH_SERVICE) as? BluetoothManager
        bluetoothAdapter = btManager?.adapter

        // ---- Method Channel ----
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, METHOD_CHANNEL)
            .setMethodCallHandler { call, result ->
                try {
                    when (call.method) {
                        "isEnabled" -> {
                            result.success(bluetoothAdapter?.isEnabled == true)
                        }
                        "getBondedDevices" -> {
                            val devices = bluetoothAdapter?.bondedDevices?.map { d ->
                                mapOf("name" to (d.name ?: ""), "address" to d.address)
                            } ?: emptyList()
                            result.success(devices)
                        }
                        "startDiscovery" -> {
                            startBtDiscovery(result)
                        }
                        "stopDiscovery" -> {
                            stopBtDiscovery()
                            result.success(null)
                        }
                        "connect" -> {
                            val address = call.argument<String>("address")
                            if (address == null) {
                                result.error("INVALID", "address is required", null)
                            } else {
                                connectToDevice(address, result)
                            }
                        }
                        "disconnect" -> {
                            disconnectDevice()
                            result.success(null)
                        }
                        "isConnected" -> {
                            result.success(bluetoothSocket?.isConnected == true)
                        }
                        "requestEnable" -> {
                            if (bluetoothAdapter?.isEnabled == true) {
                                result.success(true)
                            } else {
                                enableBtResult = result
                                val enableIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
                                enableBtLauncher.launch(enableIntent)
                            }
                        }
                        "openBluetoothSettings" -> {
                            val intent = Intent(Settings.ACTION_BLUETOOTH_SETTINGS)
                            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
                            startActivity(intent)
                            result.success(null)
                        }
                        else -> result.notImplemented()
                    }
                } catch (e: SecurityException) {
                    result.error("PERMISSION", "Bluetooth permission denied: ${e.message}", null)
                } catch (e: Exception) {
                    result.error("ERROR", e.message, null)
                }
            }

        // ---- Data EventChannel (incoming bytes from ESP32) ----
        EventChannel(flutterEngine.dartExecutor.binaryMessenger, DATA_EVENT_CHANNEL)
            .setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                    dataEventSink = events
                    startReadThread()
                }
                override fun onCancel(arguments: Any?) {
                    stopReadThread()
                    dataEventSink = null
                }
            })

        // ---- Discovery EventChannel ----
        EventChannel(flutterEngine.dartExecutor.binaryMessenger, DISCOVERY_EVENT_CHANNEL)
            .setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                    discoveryEventSink = events
                }
                override fun onCancel(arguments: Any?) {
                    stopBtDiscovery()
                    discoveryEventSink = null
                }
            })
    }

    // ====== Discovery ======

    private fun startBtDiscovery(result: MethodChannel.Result) {
        try {
            // Register receiver
            val filter = IntentFilter().apply {
                addAction(BluetoothDevice.ACTION_FOUND)
                addAction(BluetoothAdapter.ACTION_DISCOVERY_FINISHED)
            }
            registerReceiver(discoveryReceiver, filter)

            // Cancel existing discovery
            bluetoothAdapter?.cancelDiscovery()
            val started = bluetoothAdapter?.startDiscovery() == true
            Log.d(TAG, "Discovery started: $started")
            result.success(started)
        } catch (e: SecurityException) {
            result.error("PERMISSION", "Bluetooth scan permission denied", null)
        }
    }

    private fun stopBtDiscovery() {
        try {
            bluetoothAdapter?.cancelDiscovery()
            unregisterReceiver(discoveryReceiver)
        } catch (_: Exception) { }
    }

    // ====== Connection ======

    private fun connectToDevice(address: String, result: MethodChannel.Result) {
        Thread {
            try {
                // Cancel discovery before connecting
                try { bluetoothAdapter?.cancelDiscovery() } catch (_: Exception) {}

                val device = bluetoothAdapter?.getRemoteDevice(address)
                if (device == null) {
                    mainHandler.post { result.error("NOT_FOUND", "Device not found: $address", null) }
                    return@Thread
                }

                Log.d(TAG, "Connecting to ${device.name} ($address)...")

                // Close existing connection
                disconnectDevice()

                val socket = device.createRfcommSocketToServiceRecord(SPP_UUID)
                socket.connect()
                bluetoothSocket = socket
                inputStream = socket.inputStream

                Log.d(TAG, "Connected to ${device.name}")
                mainHandler.post { result.success(true) }
            } catch (e: IOException) {
                Log.e(TAG, "Connection failed: ${e.message}")
                mainHandler.post { result.error("CONNECT_FAILED", "Connection failed: ${e.message}", null) }
            } catch (e: SecurityException) {
                mainHandler.post { result.error("PERMISSION", "Permission denied: ${e.message}", null) }
            }
        }.start()
    }

    private fun disconnectDevice() {
        stopReadThread()
        try { inputStream?.close() } catch (_: Exception) {}
        try { bluetoothSocket?.close() } catch (_: Exception) {}
        inputStream = null
        bluetoothSocket = null
        Log.d(TAG, "Disconnected")
    }

    // ====== Read Thread ======

    private fun startReadThread() {
        if (isReading) return
        val stream = inputStream ?: return

        isReading = true
        readThread = Thread {
            val buffer = ByteArray(1024)
            Log.d(TAG, "Read thread started")
            while (isReading) {
                try {
                    val bytesRead = stream.read(buffer)
                    if (bytesRead > 0) {
                        val data = buffer.copyOf(bytesRead)
                        // Send raw bytes as List<Int> to Dart
                        mainHandler.post {
                            dataEventSink?.success(data.map { it.toInt() and 0xFF })
                        }
                    }
                } catch (e: IOException) {
                    if (isReading) {
                        Log.e(TAG, "Read error (disconnected?): ${e.message}")
                        mainHandler.post {
                            dataEventSink?.error("DISCONNECTED", "Device disconnected", null)
                        }
                    }
                    break
                }
            }
            Log.d(TAG, "Read thread stopped")
        }
        readThread?.isDaemon = true
        readThread?.start()
    }

    private fun stopReadThread() {
        isReading = false
        readThread?.interrupt()
        readThread = null
    }

    override fun onDestroy() {
        disconnectDevice()
        try { unregisterReceiver(discoveryReceiver) } catch (_: Exception) {}
        super.onDestroy()
    }
}
