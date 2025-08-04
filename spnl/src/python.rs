use pyo3::prelude::*;

#[cfg(feature = "run_py")]
fn handle_run_err(e: crate::run::result::SpnlError) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!("Error in generation {e}"))
}

#[cfg(feature = "run_py")]
fn handle_serde_err(e: serde_json::Error) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!("Error in deserialization {e}"))
}

#[cfg(feature = "run_py")]
#[pyclass]
#[derive(Debug, Clone)]
pub struct UsageStats {
    #[pyo3(get)]
    pub prompt_tokens: usize,

    #[pyo3(get)]
    pub completion_tokens: usize,

    #[pyo3(get)]
    total_tokens: usize,

    #[pyo3(get)]
    cost_usd: Option<f32>,
}

#[cfg(feature = "run_py")]
#[pyclass]
#[derive(Debug)]
pub struct ChatResponse {
    #[pyo3(get)]
    pub data: String,

    #[pyo3(get, set)]
    pub model_id: Option<String>,

    #[pyo3(get)]
    pub usage: Option<UsageStats>,
}

#[cfg(feature = "run_py")]
#[pyfunction]
pub async fn execute(q: String) -> Result<ChatResponse, PyErr> {
    let query: crate::Query = serde_json::from_str(q.as_str()).map_err(handle_serde_err)?;

    let rt = tokio::runtime::Runtime::new()?;
    let res = rt.block_on(crate::run::run(
        &query,
        &crate::run::RunParameters {
            vecdb_uri: "".into(),
            vecdb_table: "".into(),
            prepare: None,
        },
        None,
    ));

    res.map(|res| ChatResponse {
        data: res.to_string(),
        model_id: None,
        usage: None,
    })
    .map_err(handle_run_err)
}

#[pymodule(name = "spnl")]
pub fn spnl_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "run_py")]
    m.add_function(wrap_pyfunction!(crate::python::execute, m)?)?;
    #[cfg(feature = "run_py")]
    m.add_class::<crate::python::ChatResponse>()?;

    #[cfg(feature = "tok")]
    m.add_class::<crate::tokenizer::TokenizedQuery>()?;

    #[cfg(feature = "tok")]
    m.add_function(wrap_pyfunction!(crate::tokenizer::tokenize_query, m)?)?;

    #[cfg(feature = "tok")]
    m.add_function(wrap_pyfunction!(crate::tokenizer::tokenize_prepare, m)?)?;

    #[cfg(feature = "tok")]
    m.add_function(wrap_pyfunction!(crate::tokenizer::init, m)?)?;

    //m.add_class::<SimpleQuery>()?;
    //m.add_class::<SimpleGenerate>()?;
    //m.add_class::<SimpleGenerateInput>()?;
    //m.add_class::<SimpleMessage>()?;

    Ok(())
}

/* TODO
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
*/
