
use std::fs;
use std::io;
use std::path::{PathBuf};
use std::env;

pub struct AppPaths {
    pub root: PathBuf,
    pub models: PathBuf,
    pub stt: PathBuf,
    pub llm: PathBuf,
    pub tts: PathBuf,
}

pub const APP_NAME: &str = "reva";

pub fn setup_app_folders() -> io::Result<AppPaths> {
    
    let base_path = dirs::data_local_dir()
        .or_else(|| env::current_dir().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Could not determine a storage location"))?;

    // 2. Construct Paths
    let root = base_path.join(format!(".{}", APP_NAME));
    let models = root.join("models");
    
    let paths = AppPaths {
        stt: models.join("stt"),
        llm: models.join("llm"),
        tts: models.join("tts"),
        models: models,
        root: root,   
    };

    // 3. Create Directories
    // We only need to call create_dir_all on the "deepest" leaf nodes; 
    // it will automatically create 'root' and 'models' as parents.
    fs::create_dir_all(&paths.root)?;
    fs::create_dir_all(&paths.models)?;
    fs::create_dir_all(&paths.stt)?;
    fs::create_dir_all(&paths.llm)?;
    fs::create_dir_all(&paths.tts)?;

    Ok(paths)
}