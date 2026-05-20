use crate::settings::AppPaths;
use std::fs::{self, File};
use std::io::{self, Read, Write, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;
use serde_json::json;

const WHISPER_BASE: &str = "https://huggingface.co/Systran/faster-whisper-base.en/resolve/main";
const WHISPER_FILES: &[&str] = &[
    "model.bin",
    "config.json",
    "tokenizer.json",
    "vocabulary.txt",
];
const LLM_MODEL_BASE: &str = "https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main";
pub const LLM_MODEL_FILE: &str = "LFM2.5-350M-Q4_K_M.gguf";

pub fn download_models(app_dir: &AppPaths) -> io::Result<()> {
    
    let agent = ureq::Agent::new_with_defaults();
    for &name in WHISPER_FILES {
        let dest = app_dir.stt.join(name);
        if dest.exists() {
            continue;
        }
        let url = format!("{}/{}", WHISPER_BASE, name);
        download(&agent, &url, &dest)?;
    }
    ensure_whisper_preprocessor_config(&app_dir.stt)?;

    let llm_model = app_dir.llm.join(&LLM_MODEL_FILE);
    let llm_url = format!("{}/{}", LLM_MODEL_BASE, LLM_MODEL_FILE);
    if !llm_model.exists(){
        download(&agent, &llm_url, &llm_model)?;
    }
    Ok(())
}

pub fn ensure_whisper_preprocessor_config(stt_dir: &PathBuf) -> io::Result<()> {
    let config_path = stt_dir.join("preprocessor_config.json");
    if config_path.exists() {
        return Ok(());
    }

    let config = json!({
        "chunk_length": 30,
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": 80,
        "hop_length": 160,
        "n_fft": 400,
        "n_samples": 480000,
        "nb_max_frames": 3000,
        "padding_side": "right",
        "padding_value": 0.0,
        "processor_class": "WhisperProcessor",
        "return_attention_mask": false,
        "sampling_rate": 16000
    });

    let json_str = serde_json::to_string_pretty(&config)?;
    fs::write(&config_path, json_str)?;
    eprintln!("Generated preprocessor_config.json at {}", config_path.display());
    Ok(())
}

fn download(agent: &ureq::Agent, url: &str, dest: &Path) -> io::Result<()> {
    eprintln!("Downloading: {}", url);
    let resp = agent
        .get(url)
        .header("User-Agent", "rust-whisp/0.1")
        .call()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{e}")))?;

    let total_len: Option<u64> = resp
        .headers()
        .get("content-length")
        .and_then(|hv| hv.to_str().ok())
        .and_then(|s| s.parse().ok());

    let tmp = dest.with_extension("part");
    let mut file = File::create(&tmp)?;
    let started = Instant::now();

    let mut body = resp.into_body();
    let mut reader = BufReader::new(body.as_reader());
    let mut chunk = vec![0u8; 256 * 1024];
    let mut copied: u64 = 0;
    let mut last_logged = 0u64;

    loop {
        let n = reader.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        file.write_all(&chunk[..n])?;
        copied += n as u64;
        if copied - last_logged >= 8 * 1024 * 1024 {
            display_progress(copied, total_len, started);
            last_logged = copied;
        }
    }

    display_progress(copied, total_len, started);
    eprintln!();
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp, dest)?;
    Ok(())
}

fn display_progress(copied: u64, total_len: Option<u64>, started: Instant) {
    let mb = copied as f64 / (1024.0 * 1024.0);
    let secs = started.elapsed().as_secs_f64().max(1e-3);
    let mbps = mb / secs;

    match total_len {
        Some(total) => {
            let total_mb = total as f64 / (1024.0 * 1024.0);
            let pct = (copied as f64 / total as f64) * 100.0;
            let bar_width = 30;
            let filled = (pct as usize * bar_width / 100).min(bar_width);
            let bar = "█".repeat(filled) + &"░".repeat(bar_width - filled);
            eprint!(
                "\r  [{}] {:.0}% | {:.1} MB / {:.1} MB | {:.1} MB/s",
                bar, pct, mb, total_mb, mbps
            );
        }
        None => {
            eprint!("\r  ⤓ {:.1} MB | {:.1} MB/s", mb, mbps);
        }
    }
    let _ = io::stderr().flush();
}