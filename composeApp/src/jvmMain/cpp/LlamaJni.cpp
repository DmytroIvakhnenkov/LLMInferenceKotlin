#include <jni.h>
#include <string>
#include <cstring>
#include <vector>
#include <cstdio>
#include <iostream>
#include "llama.h" // The llama.cpp header

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_example_project_LlamaJni_loadCtx(JNIEnv* env, jobject /*obj*/, jstring path) {
    const char* cPath = env->GetStringUTFChars(path, nullptr);

    ggml_backend_load_all();

    struct llama_model_params params = llama_model_default_params();
    // Example config
    params.n_gpu_layers = 99;
    params.use_mmap = true;
    params.use_mlock = false;

    struct llama_model* model = llama_model_load_from_file(cPath, params);
    if (!model) {
        fprintf(stderr, "Failed to load model from %s\n", cPath);
        env->ReleaseStringUTFChars(path, cPath);
        return 0;
    }

    struct llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx = 40960;
    ctx_params.n_batch = 40960;
    ctx_params.no_perf = false;


    struct llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        env->ReleaseStringUTFChars(path, cPath);
        return 0;
    }

    env->ReleaseStringUTFChars(path, cPath);

    // Return the pointer as a jlong so we can store it in Kotlin
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT jstring JNICALL
Java_org_example_project_LlamaJni_generateNextToken(JNIEnv* env, jobject /*obj*/, jlong ctxPointer, jstring promptJ) {
    struct llama_context* ctx = reinterpret_cast<struct llama_context*>(ctxPointer);
    if (!ctx) {
        fprintf(stderr, "Invalid context pointer\n");
        return env->NewStringUTF("");
    }

    const llama_model* model = llama_get_model(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);

    const char* prompt_cstr = env->GetStringUTFChars(promptJ, nullptr);
    std::string prompt(prompt_cstr);

    // ------------------------
    // 6. Initialize the token sampler
    // ------------------------
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // ------------------------
    // 7. Define helper function to generate response from a prompt
    // ------------------------
    auto generate = [&](const std::string & prompt) {
        std::string response;

        const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;

        // Tokenize prompt
        const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
            GGML_ABORT("failed to tokenize the prompt\n");
        }

        // Prepare batch
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        llama_token new_token_id;

        while (true) {
            // Check context space
            int n_ctx = llama_n_ctx(ctx);
            int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            if (n_ctx_used + batch.n_tokens > n_ctx) {
                printf("\033[0m\n");
                fprintf(stderr, "context size exceeded\n");
                exit(0);
            }

            // Decode batch
            int ret = llama_decode(ctx, batch);
            if (ret != 0) {
                GGML_ABORT("failed to decode, ret = %d\n", ret);
            }

            // Sample next token
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // Stop if end-of-generation token
            if (llama_vocab_is_eog(vocab, new_token_id)) break;

            // Convert token to string, print and append
            char buf[256];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) GGML_ABORT("failed to convert token to piece\n");
            std::string piece(buf, n);
            response += piece;

            // Prepare next batch
            batch = llama_batch_get_one(&new_token_id, 1);
        }

        return response;
    };


    std::vector<llama_chat_message> messages;
        std::vector<char> formatted(llama_n_ctx(ctx));
        int prev_len = 0;

        // Get chat template from model
        const char * tmpl = llama_model_chat_template(model, /* name */ nullptr);

        // Add user message to conversation
        messages.push_back({"user", strdup(prompt_cstr)});

        // Apply template to get prompt
        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        }
        if (new_len < 0) {
            fprintf(stderr, "failed to apply the chat template\n");
            return env->NewStringUTF("");
        }

        // Prepare prompt from formatted messages
        std::string user_prompt(formatted.begin() + prev_len, formatted.begin() + new_len);

        // Generate assistant response
        printf("\033[33m");
        std::string response = generate(user_prompt);
        printf("\n\033[0m");



    env->ReleaseStringUTFChars(promptJ, prompt_cstr);

    llama_sampler_free(smpl);

    // Return the pointer as a jlong so we can store it in Kotlin
    return env->NewStringUTF(response.c_str());
}

}