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
import org.jetbrains.compose.ui.tooling.preview.Preview

@Composable
@Preview
fun App() {
    MaterialTheme {
        var inputText by remember { mutableStateOf(TextFieldValue("")) }

        // Track messages with sender info
        val messages = remember { mutableStateListOf<Message>() }

        Column(
            modifier = Modifier
                .background(MaterialTheme.colorScheme.primaryContainer)
                .safeContentPadding()
                .fillMaxSize()
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
                    placeholder = { Text("Enter message") }
                )
                Spacer(modifier = Modifier.width(8.dp))
                Button(
                    onClick = {
                        if (inputText.text.isNotBlank()) {
                            // Add user message
                            messages.add(Message(inputText.text, Sender.USER))
                            // Add LLM response
                            messages.add(Message(dummySendFunction(inputText.text), Sender.LLM))
                            inputText = TextFieldValue("") // Clear input
                        }
                    }
                ) {
                    Text("Send")
                }
            }

            // Messages list
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.Top
            ) {
                for (message in messages) {
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
                            Text(message.text, color = Color.Black)
                        }
                    }
                }
            }
        }
    }
}

// Message sender enum
enum class Sender {
    USER, LLM
}

// Message model
data class Message(val text: String, val sender: Sender)

// Dummy LLM function
fun dummySendFunction(input: String): String {
    return "Hello, I am Koli"
}
