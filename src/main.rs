mod download;
mod chat;
mod settings;

use download::{download_models, LLM_MODEL_FILE};
use chat::{start_chat};
use settings::{setup_app_folders};

fn main() -> std::io::Result<()> {

    let app_paths = setup_app_folders()?;
    let _get_stt_models = download_models(&app_paths);
    let default_llm_path = app_paths.llm.join(LLM_MODEL_FILE).to_str().unwrap().to_string();
    let llm_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| default_llm_path);
    let _init = start_chat(llm_path);
    Ok(())
}