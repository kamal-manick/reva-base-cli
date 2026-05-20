#$env:RUSTFLAGS="-C target-feature=+crt-static"
$env:LIBCLANG_PATH = "C:\Custom\clang\bin";
cargo build --release