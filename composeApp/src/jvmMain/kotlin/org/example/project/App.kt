
package org.example.project

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import org.example.project.LlamaJni.generateNextToken
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.runtime.snapshots.SnapshotStateList
import org.jetbrains.compose.ui.tooling.preview.Preview
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.consumeEach
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.plus
import kotlin.collections.plus

// Message sender enum
enum class Sender {
    USER, LLM
}

// Message model
data class Message(val text: String, val sender: Sender)


@Composable
@Preview
fun App() {
    MaterialTheme {
        Row(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
        ) {
            SettingsPanel()
            ChatPanel()
        }
    }
}


@Composable
fun SettingsPanel(
) {
    Column(
        modifier = Modifier
            .width(200.dp)
            .fillMaxHeight()
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(8.dp)
    ) {
        Text("Settings", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(8.dp))

        Button(onClick = {  }) {
            Text("Option 1")
        }
        Button(onClick = {  }) {
            Text("Option 2")
        }
    }
}

@Composable
fun ChatPanel(){

    var inputText by remember { mutableStateOf(TextFieldValue("")) }
    var messages by remember { mutableStateOf(listOf<Message>()) }
    var isStreaming by remember { mutableStateOf(false) }

    // Key that only changes when user sends a message, not when LLM messages update
    val userMessageCount = messages.count { it.sender == Sender.USER }

    // This LaunchedEffect only triggers when a new USER message is added
    LaunchedEffect(userMessageCount) {
        // Only start streaming if we have user messages and we're not already streaming
        val lastUserMessage = messages.lastOrNull { it.sender == Sender.USER }
        if (lastUserMessage != null && !isStreaming) {
            isStreaming = true

            // Add empty LLM message that will be updated
            messages = messages + Message("", Sender.LLM)
            val llmMessageIndex = messages.lastIndex

            try {
                var accumulatedText = ""
                streamTokensBufferedV2(lastUserMessage.text, paceMs = 50).collect { token ->
                    accumulatedText += token

                    // Update the LLM message in place
                    messages = messages.toMutableList().also { messageList ->
                        if (llmMessageIndex < messageList.size) {
                            messageList[llmMessageIndex] = Message(accumulatedText, Sender.LLM)
                        }
                    }
                }
            } catch (e: Exception) {
                println("Streaming error: ${e.message}")
                // Optionally update the message with error info
                messages = messages.toMutableList().also { messageList ->
                    if (llmMessageIndex < messageList.size) {
                        messageList[llmMessageIndex] = Message("Error: ${e.message}", Sender.LLM)
                    }
                }
            } finally {
                isStreaming = false
            }
        }
    }

    Column(
        modifier = Modifier
            .background(MaterialTheme.colorScheme.primaryContainer)
            .safeContentPadding()
            .fillMaxHeight()
            .padding(16.dp)
    ) {

        // Input Row
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            TextField(
                value = inputText,
                onValueChange = { inputText = it },
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(20.dp)),
                placeholder = { Text("Enter message") },
                enabled = !isStreaming // Disable input while streaming
            )
            Spacer(modifier = Modifier.width(8.dp))
            Button(
                onClick = {
                    if (inputText.text.isNotBlank() && !isStreaming) {
                        messages = messages + Message(inputText.text, Sender.USER)
                        inputText = TextFieldValue("")
                    }
                },
                enabled = !isStreaming && inputText.text.isNotBlank()
            ) {
                if (isStreaming) {
                    Text("Streaming...")
                } else {
                    Text("Send")
                }
            }
        }

        val listState = rememberLazyListState()

        // Messages list
        LazyColumn(
            modifier = Modifier.fillMaxWidth().weight(1f),
            state = listState,
        ) {
            items(messages, key = { "${it.sender}-${messages.indexOf(it)}" }) { message ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = if (message.sender == Sender.USER) Arrangement.End else Arrangement.Start
                ) {
                    Box(
                        modifier = Modifier
                            .padding(vertical = 4.dp)
                            .clip(RoundedCornerShape(20.dp))
                            .background(
                                if (message.sender == Sender.USER) Color(0xFF64B5F6) // blue
                                else Color(0xFFB2FF59) // green
                            )
                            .padding(horizontal = 16.dp, vertical = 10.dp)
                    ) {
                        Text(
                            text = if (message.text.isEmpty() && isStreaming) "..." else message.text,
                            color = Color.Black
                        )
                    }
                }
            }
        }

        // Automatically scroll to the bottom when messages are updated
        LaunchedEffect(messages.lastOrNull()) {
            if (messages.isNotEmpty()) {
                listState.animateScrollToItem(messages.lastIndex, scrollOffset = Int.MAX_VALUE)
            }
        }
    }
}

// Improved streaming function with independent coroutine scope
fun streamTokensBufferedV2(prompt: String, paceMs: Long = 50): Flow<String> = callbackFlow {
    val tokenBuffer = Channel<String>(Channel.UNLIMITED)

    // Use independent scope that won't be cancelled by Compose recomposition
    val independentScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    // Launch JNI generation in independent scope
    val generationJob = independentScope.launch {
        try {
            println("Starting JNI generation for prompt: $prompt")
            LlamaJni.generateNextTokenStream(LlamaJni.ctxPointer, prompt) { token: String ->
                val result = tokenBuffer.trySend(token)
                if (result.isFailure) {
                    println("Failed to send token to buffer: ${result.exceptionOrNull()}")
                }
            }
            println("JNI generation completed")
        } catch (e: Exception) {
            println("Error in JNI generation: ${e.message}")
            // Send error to flow
            trySend("Error: ${e.message}")
        } finally {
            // Send completion signal
            println("Closing token buffer")
            tokenBuffer.close()
        }
    }

    // Consume tokens with controlled pace in independent scope
    val consumerJob = independentScope.launch {
        try {
            tokenBuffer.consumeEach { token ->
                // Use channel's trySend instead of flow's trySend to avoid compose scope issues
                val sent = trySend(token)
                if (sent.isFailure) {
                    println("Failed to emit token: ${sent.exceptionOrNull()}")
                }
                delay(paceMs)
            }
            println("Token consumption completed")
        } catch (e: Exception) {
            println("Error in token consumption: ${e.message}")
        } finally {
            // Close the flow when buffer is consumed
            close()
        }
    }

    awaitClose {
        println("Cleaning up streaming resources")
        generationJob.cancel()
        consumerJob.cancel()
        independentScope.cancel()
        if (!tokenBuffer.isClosedForSend) {
            tokenBuffer.close()
        }
    }
}