#include <jni.h>
#include <string>
#include <vector>
#include <cstdio>
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

    ctx_params.n_ctx = 1000;
    ctx_params.n_batch = 1000;
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
    env->ReleaseStringUTFChars(promptJ, prompt_cstr);

    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        fprintf(stderr, "Failed to tokenize prompt (n_prompt=%d)\n", n_prompt);
        return env->NewStringUTF("");
    }

    std::vector<llama_token> prompt_tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);
     // Sampler
    llama_sampler* smpl = llama_sampler_init_greedy();

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // number of tokens to predict
    int n_predict = 32;
    std::string generated_text;


    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return env->NewStringUTF("");
        }

        n_pos += batch.n_tokens;

        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "Error converting token to piece\n");
            break;
        }
        generated_text.append(buf, n);

        batch = llama_batch_get_one(&new_token_id, 1);
    }


    llama_sampler_free(smpl);

    // Return the pointer as a jlong so we can store it in Kotlin
    return env->NewStringUTF(generated_text.c_str());
}

}