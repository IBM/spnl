use wasm_bindgen::prelude::*;

use spnl::{PlanOptions, from_yaml_str, plan};

/*#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}*/

#[wasm_bindgen]
pub async fn compile_query(query: &str) -> Result<String, JsError> {
    let program = plan(
        &from_yaml_str(query)?,
        &PlanOptions {
            max_aug: Some(10),
            vecdb_uri: "".into(),
            vecdb_table: "".into(),
        },
    )
    .await
    .map_err(|e| JsError::new(e.to_string().as_str()))?;

    //Ok(serde_wasm_bindgen::to_value(&program)?)
    Ok(serde_json::to_string(&program)?)
}
