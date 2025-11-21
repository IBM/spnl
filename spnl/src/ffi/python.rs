use pyo3::prelude::*;

#[cfg(feature = "run_py")]
fn handle_execute_err(e: crate::SpnlError) -> PyErr {
    pyo3::exceptions::PyOSError::new_err(format!("{e}"))
}

#[cfg(feature = "run_py")]
fn handle_serde_err(e: serde_json::Error) -> PyErr {
    pyo3::exceptions::PyOSError::new_err(format!("Error in deserialization {e}"))
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
    let query: crate::ir::Query = serde_json::from_str(q.as_str()).map_err(handle_serde_err)?;

    let rt = tokio::runtime::Runtime::new()?;
    let res = rt.block_on(crate::execute(
        &query,
        &crate::ExecuteOptions { prepare: None },
    ));

    res.map(|res| ChatResponse {
        data: res.to_string(),
        model_id: None,
        usage: None,
    })
    .map_err(handle_execute_err)
}

#[pymodule(name = "spnl")]
pub fn spnl_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // TODO pyo3::create_exception!(m, PyModelNotFoundError, pyo3::exceptions::PyException);

    #[cfg(feature = "run_py")]
    m.add_function(wrap_pyfunction!(crate::ffi::python::execute, m)?)?;
    #[cfg(feature = "run_py")]
    m.add_class::<crate::ffi::python::ChatResponse>()?;

    #[cfg(feature = "tok")]
    m.add_class::<crate::optimizer::llo::tokenize::TokenizedQuery>()?;

    #[cfg(feature = "tok")]
    m.add_class::<crate::optimizer::llo::tokenize::TokenizedChatCompletionQuery>()?;

    #[cfg(feature = "tok")]
    m.add_class::<crate::optimizer::llo::tokenize::CompletionRequest>()?;

    #[cfg(feature = "tok")]
    m.add_function(wrap_pyfunction!(
        crate::optimizer::llo::tokenize::tokenize_query,
        m
    )?)?;

    #[cfg(feature = "tok")]
    m.add_function(wrap_pyfunction!(
        crate::optimizer::llo::tokenize::tokenize_prepare,
        m
    )?)?;

    #[cfg(feature = "tok")]
    m.add_function(wrap_pyfunction!(crate::optimizer::llo::tokenize::init, m)?)?;

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
