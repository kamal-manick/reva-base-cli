use anyhow::Result;
use encoding_rs::UTF_8;
use std::io::{self, Write};
use std::num::NonZeroU32;
use std::path::{Path};

use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel},
    sampling::LlamaSampler,
    token::LlamaToken,
    gguf::GgufContext,
};

/// A simple multi-turn chat session that keeps history and streams tokens as generated.
///
/// `model` is heap-allocated in a `Box` so its address is stable while `ChatSession` moves.
/// `ctx` and `batch` use `'static` lifetimes via an unsafe transmute — this is sound because
/// both are dropped before `model` (field declaration order), and we never move `model` out.
struct ChatSession {
    backend: LlamaBackend,
    model: Box<LlamaModel>,
    tmpl: LlamaChatTemplate,
    // Option so we can drop the old context before creating a new one during rebuild.
    ctx: Option<llama_cpp_2::context::LlamaContext<'static>>,
    batch: LlamaBatch<'static>,
    sampler: LlamaSampler,

    // streaming-safe UTF-8 decoder for token pieces
    decoder: encoding_rs::Decoder,

    // conversation history in chat-message form
    history: Vec<LlamaChatMessage>,

    // prompt tokens currently loaded into ctx (so we can detect when we need to rebuild)
    loaded_prompt_tokens: Vec<LlamaToken>,

    // track current position (number of tokens already in ctx)
    pos: usize,

    // configured context window for simple overflow handling
    n_ctx: usize,
}

impl ChatSession {
    fn new(model_path: &str, n_ctx: usize, temperature: f32, top_p: f32) -> Result<Self> {
        // Backend
        let mut backend = LlamaBackend::init()?;
        backend.void_logs(); // suppress llama.cpp logs

        // Model — boxed for stable heap address
        let model = Box::new(LlamaModel::load_from_file(&backend, model_path, &LlamaModelParams::default())?);

        // Context
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(n_ctx as u32).unwrap()));
        
        // SAFETY: model is heap-allocated and outlives ctx; we never move model out of the Box.
        let ctx = unsafe { make_ctx(&model, &backend, ctx_params)? };

        // Chat template (from model)
        let tmpl = model.chat_template(None)?; // baked-in default template

        // Batch — LlamaBatch::new borrows nothing, so 'static is trivially valid.
        let batch: LlamaBatch<'static> = LlamaBatch::new(512, 1);

        let seed = 42u32;
        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.10, 0.0, 0.0),
            LlamaSampler::temp(temperature),
            LlamaSampler::top_p(top_p, 1),
            LlamaSampler::dist(seed),
        ]);

        // UTF-8 decoder for token_to_piece streaming
        let decoder = UTF_8.new_decoder();

        Ok(Self {
            backend,
            model,
            tmpl,
            ctx: Some(ctx),
            batch,
            sampler,
            decoder,
            history: Vec::new(),
            loaded_prompt_tokens: Vec::new(),
            pos: 0,
            n_ctx,
        })
    }

    /// Add a system prompt once at the beginning.
    fn set_system(&mut self, system: &str) -> Result<()> {
        self.history.clear();
        self.loaded_prompt_tokens.clear();
        self.pos = 0;

        self.history
            .push(LlamaChatMessage::new("system".to_string(), system.to_string())?);

        // Build initial prompt (ends with assistant tag)
        self.rebuild_context_from_history(true)?;
        Ok(())
    }

    /// Main API you'd call from an HTTP endpoint:
    /// - adds user message
    /// - generates assistant reply (streaming tokens)
    /// - appends assistant message to history
    fn chat(&mut self, user_text: &str, max_new_tokens: Option<usize>) -> Result<String> {
        self.history
            .push(LlamaChatMessage::new("user".to_string(), user_text.to_string())?);

        // Ensure ctx has the latest prompt for current history, ending with assistant tag.
        self.ensure_prompt_loaded()?;

        // Generate assistant tokens
        let mut assistant_out = String::new();

        let mut generated = 0usize;
        loop {
            if let Some(limit) = max_new_tokens {
                if generated >= limit {
                    break;
                }
            }

            // Sample & accept a token from logits
            // Access fields directly so Rust sees disjoint borrows (sampler vs ctx).
            let token = self.sampler.sample(self.ctx.as_ref().unwrap(), -1);

            // Stop on EOG tokens (end-of-turn/end-of-seq etc.).
            if self.model.is_eog_token(token) {
                break;
            }

            // Convert token -> text piece, stream immediately
            let piece = self
                .model
                .token_to_piece(token, &mut self.decoder, false, None)?;

            print!("{}", piece);
            io::stdout().flush()?;

            assistant_out.push_str(&piece);

            // Feed token back into ctx (decode)
            self.batch.clear();
            self.batch.add(token, self.pos as i32, &[0], true)?;
            // Split borrows: ctx and batch are disjoint fields.
            self.ctx.as_mut().unwrap().decode(&mut self.batch)?;

            self.pos += 1;
            generated += 1;

            if self.pos + 8 >= self.n_ctx {
                eprintln!("\n[warn] context window nearly full; stopping generation for this turn");
                break;
            }
        }

        println!();

        // Add assistant message to history — context already contains all decoded tokens,
        // so no rebuild needed here.
        self.history.push(LlamaChatMessage::new(
            "assistant".to_string(),
            assistant_out.clone(),
        )?);
        self.loaded_prompt_tokens.clear(); // mark stale so ensure_prompt_loaded resyncs next turn

        Ok(assistant_out)
    }

    fn ensure_prompt_loaded(&mut self) -> Result<()> {
        let prompt = self.model.apply_chat_template(&self.tmpl, &self.history, true)?;

        let new_tokens = if self.loaded_prompt_tokens.is_empty() {
            self.model.str_to_token(&prompt, AddBos::Always)?
        } else {
            self.model.str_to_token(&prompt, AddBos::Never)?
        };

        let mut lcp = 0usize;
        while lcp < self.loaded_prompt_tokens.len()
            && lcp < new_tokens.len()
            && self.loaded_prompt_tokens[lcp] == new_tokens[lcp]
        {
            lcp += 1;
        }

        if lcp == self.loaded_prompt_tokens.len() {
            let delta = &new_tokens[lcp..];
            if !delta.is_empty() {
                self.decode_tokens_into_ctx(delta)?;
                self.loaded_prompt_tokens = new_tokens;
            }
            Ok(())
        } else {
            self.rebuild_context_from_tokens(&new_tokens)?;
            Ok(())
        }
    }

    fn rebuild_context_from_history(&mut self, first_time: bool) -> Result<()> {
        let prompt = self.model.apply_chat_template(&self.tmpl, &self.history, true)?;
        let tokens = self.model.str_to_token(
            &prompt,
            if first_time { AddBos::Always } else { AddBos::Always },
        )?;
        self.rebuild_context_from_tokens(&tokens)?;
        Ok(())
    }

    fn rebuild_context_from_tokens(&mut self, tokens: &[LlamaToken]) -> Result<()> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(self.n_ctx as u32).unwrap()));

        // Drop old context first so there's no active borrow on model through the old ctx.
        self.ctx = None;

        // SAFETY: model is heap-allocated and outlives ctx; we never move model out of the Box.
        let new_ctx = unsafe { make_ctx(&self.model, &self.backend, ctx_params)? };
        self.ctx = Some(new_ctx);

        self.batch.clear();
        self.pos = 0;
        self.loaded_prompt_tokens = tokens.to_vec();

        let seed = 42u32;
        self.sampler = self.sampler_from_defaults(seed);

        self.decode_tokens_into_ctx(tokens)?;

        Ok(())
    }

    fn sampler_from_defaults(&self, seed: u32) -> LlamaSampler {
        LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.10, 0.0, 0.0),
            LlamaSampler::temp(0.7),
            LlamaSampler::top_p(0.9, 1),
            LlamaSampler::dist(seed),
        ])
    }

    fn decode_tokens_into_ctx(&mut self, tokens: &[LlamaToken]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }

        self.sampler.accept_many(tokens.iter());

        let mut idx = 0usize;
        while idx < tokens.len() {
            self.batch.clear();
            let end = usize::min(idx + 512, tokens.len());
            let last = end - 1;

            for i in idx..end {
                let want_logits = i == last;
                self.batch
                    .add(tokens[i], (self.pos + (i - idx)) as i32, &[0], want_logits)?;
            }

            // Split borrows: ctx and batch are disjoint fields.
            self.ctx.as_mut().unwrap().decode(&mut self.batch)?;
            self.pos += end - idx;
            idx = end;

            if self.pos + 8 >= self.n_ctx {
                eprintln!("\n[warn] prompt exceeds context window; consider trimming history");
                break;
            }
        }

        Ok(())
    }
}

/// Create a `LlamaContext` and transmute its lifetime to `'static`.
///
/// SAFETY: caller must ensure `model` and `backend` outlive the returned context,
/// and that the context is dropped before `model` is dropped or moved.
unsafe fn make_ctx(
    model: &LlamaModel,
    backend: &LlamaBackend,
    ctx_params: LlamaContextParams,
) -> Result<llama_cpp_2::context::LlamaContext<'static>> {
    let ctx = model.new_context(backend, ctx_params)?;
    // SAFETY: caller guarantees model/backend outlive the returned context.
    Ok(unsafe {
        std::mem::transmute::<
            llama_cpp_2::context::LlamaContext<'_>,
            llama_cpp_2::context::LlamaContext<'static>,
        >(ctx)
    })
}

fn get_model_name(gguf: &GgufContext) -> Option<String> {
    let idx = gguf.find_key("general.name");
    if idx == -1 || gguf.kv_type(idx) != 8 {
        return None;
    }
    gguf.val_str(idx).map(|s| s.to_string())
}

fn get_context_length(gguf: &GgufContext) -> u32 {
    let fallback = 2048u32;

    // 1. Find the architecture index
    let arch_idx = gguf.find_key("general.architecture");
    if arch_idx == -1 {
        return fallback;
    }

    // 2. Get the architecture string (GGUF_TYPE_STRING is 8)
    // We check type to prevent llama.cpp internal panics
    if gguf.kv_type(arch_idx) != 8 {
        return fallback;
    }

    let arch = match gguf.val_str(arch_idx) {
        Some(s) => s,
        None => return fallback,
    };

    // 3. Construct the specific key (e.g., "qwen2.context_length")
    let context_key = format!("{}.context_length", arch);
    let ctx_idx = gguf.find_key(&context_key);

    if ctx_idx == -1 {
        return fallback;
    }

    // 4. Map the GGUF type to the correct retrieval function
    // GGUF_TYPE_UINT8: 0, UINT16: 2, UINT32: 4, INT32: 5, UINT64: 10, INT64: 11
    match gguf.kv_type(ctx_idx) {
        4 => gguf.val_u32(ctx_idx),               // UINT32
        5 => gguf.val_i32(ctx_idx) as u32,        // INT32
        10 => gguf.val_u64(ctx_idx) as u32,       // UINT64
        // Add more if you encounter weirdly typed models:
        // 0 => gguf.val_u8(ctx_idx) as u32, 
        _ => fallback,
    }
}

pub fn start_chat(model_path: String) -> Result<()> {
    let mpath = Path::new(&model_path);
    let (model_ctx_length, model_name) = if let Some(g_ctx) = GgufContext::from_file(mpath) {
        let ctx_len = get_context_length(&g_ctx);
        let name = get_model_name(&g_ctx);
        (Some(ctx_len), name)
    } else {
        eprintln!("Failed to load GGUF context from file.");
        (None, None)
    };

    let n_ctx = model_ctx_length.unwrap_or(2048) as usize;
    let temperature = 0.7;
    let top_p = 0.9;
    
    let mut session = ChatSession::new(&model_path, n_ctx, temperature, top_p)?;
    session.set_system("You are a helpful assistant. Answer clearly and concisely.")?;

    // ANSI helpers
    const RESET:     &str = "\x1b[0m";
    const BOLD:      &str = "\x1b[1m";
    const GREEN:     &str = "\x1b[32m";
    const CYAN:      &str = "\x1b[36m";
    const GREY:      &str = "\x1b[90m";
    const BLUE_BOLD: &str = "\x1b[1;34m";

    let name = model_name.as_deref().unwrap_or("LLM");
    let sep = format!("{GREY}════════════════════════════════════════{RESET}");
    println!("{sep}");
    println!("  {BOLD}{GREEN}Model{RESET}  : {CYAN}{BOLD}{name}{RESET}");
    println!("  {BOLD}{GREEN}Context{RESET}: {CYAN}{BOLD}{n_ctx} tokens{RESET}");
    println!("{sep}");
    println!("{GREY}Type your message and press Enter. Type /exit to quit.{RESET}\n");

    let mut line = String::new();
    loop {
        print!("{BLUE_BOLD}You:{RESET} ");
        io::stdout().flush()?;
        line.clear();
        io::stdin().read_line(&mut line)?;
        let user = line.trim();
        if user == "/exit" {
            break;
        }
        if user.is_empty() {
            continue;
        }

        print!("{BOLD}{GREEN}Assistant:{RESET} ");
        io::stdout().flush()?;
        let _ = session.chat(user, None)?;
        println!();
    }

    Ok(())
}