use tauri::Manager;

#[tauri::command]
fn get_platform() -> String {
    std::env::consts::OS.to_string()
}

#[tauri::command]
async fn open_devtools(window: tauri::Window) {
    #[cfg(debug_assertions)]
    window.open_devtools();
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .invoke_handler(tauri::generate_handler![get_platform, open_devtools])
        .setup(|app| {
            let window = app.get_webview_window("main").unwrap();
            // Restore window state (size/position) from previous session
            window.restore_state(tauri_plugin_window_state::StateFlags::all())?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error running NexusAI desktop");
}
