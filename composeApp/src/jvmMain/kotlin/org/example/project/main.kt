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

    val ctxPointer = loadCtx("/home/dmytro/llama.cpp/Qwen3-1.7B-Q8_0.gguf")


}

fun main() = application {

//    cleanUpLlama(modelPointer, ctxPtr)
    Window(
        onCloseRequest = ::exitApplication,
        title = "KotlinProject",
    ) {
        App()
    }
}