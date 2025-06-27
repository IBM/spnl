use pyo3::prelude::*;

#[pymodule(name = "spnl")]
pub fn spnl_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
