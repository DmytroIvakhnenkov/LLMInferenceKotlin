package org.example.project

import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import org.example.project.LlamaJni.loadCtx
import org.example.project.LlamaJni.generateNextToken




object LlamaJni {
    init {
        System.load("/home/dmytro/KotlinProject/composeApp/src/jvmMain/cpp/build/libllama_jni.so") // loads libllama_jni.so
    }

    external fun loadCtx(path: String): Long
    external fun generateNextToken(ctxPtr: Long, prompt: String): String
    external fun generateNextTokenStream(ctxPtr: Long, prompt: String, onToken: (String) -> Unit): Unit

    val ctxPointer = loadCtx("/home/dmytro/llama.cpp/Qwen3-32B-Q4_K_M.gguf")



}

fun main() = application {
    LlamaJni.ctxPointer
//    cleanUpLlama(modelPointer, ctxPtr)
    Window(
        onCloseRequest = ::exitApplication,
        title = "KotlinProject",
    ) {
        App()
    }
}