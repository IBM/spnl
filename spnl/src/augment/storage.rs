use std::sync::Arc;

use anyhow::Ok;
use arrow_array::{
    FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray,
    /*cast::AsArray,*/ types::Float32Type,
};
use arrow_schema::ArrowError;

use lancedb::query::{ExecutableQuery, QueryBase};

use futures::TryStreamExt;
use lancedb::{
    Table,
    arrow::arrow_schema::{DataType, Field, Schema},
};
use tracing::warn;

pub struct VecDB {
    default_table: Table,
}

impl VecDB {
    pub async fn connect(db_path: &str, default_table: &str) -> anyhow::Result<Self> {
        let connection = lancedb::connect(db_path).execute().await?;
        let table_exists = connection
            .table_names()
            .execute()
            .await?
            .contains(&default_table.to_string());
        if !table_exists {
            warn!("Table {} does not exist, creating it", default_table);
            let schema = Self::get_default_schema();
            connection
                .create_empty_table(default_table, schema)
                .execute()
                .await?;
        }
        let table = connection.open_table(default_table).execute().await?;
        Ok(Self {
            default_table: table,
        })
    }

    pub async fn find_similar_keys(
        &self,
        key: &str,
        vector: Vec<f32>,
        n: usize,
        range_min: Option<f32>,
        range_max: Option<f32>,
    ) -> anyhow::Result<impl DoubleEndedIterator<Item = String>> {
        use itertools::Itertools; // for .unique()
        Ok(self
            .find_similar(vector, n, range_min, range_max)
            .await?
            .into_iter()
            .filter_map(|record_batch| {
                if let Some(files_array) = record_batch.column_by_name(key)
                    && let Some(files) = files_array
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                {
                    return Some(
                        files
                            .iter()
                            .filter_map(|b| b.map(|b| b.to_string()))
                            .collect::<Vec<String>>(),
                    );
                }

                // no matching docs for this body vector
                None
            })
            .flatten()
            .unique())
    }

    pub async fn find_similar(
        &self,
        vector: Vec<f32>,
        n: usize,
        range_min: Option<f32>,
        range_max: Option<f32>,
    ) -> anyhow::Result<Vec<RecordBatch>> {
        Ok(self
            .default_table
            .query()
            .nearest_to(vector)?
            .distance_range(range_min, range_max.or(Some(1.0)))
            .limit(n)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?)
    }

    /// Get the default schema for the VecDB
    fn get_default_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("filename", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    1024,
                ),
                true,
            ),
        ]))
    }

    pub async fn add_vector<I1, I2>(
        &self,
        filenames: I1,
        vectors: I2,
        vec_dim: i32,
    ) -> anyhow::Result<()>
    where
        I1: IntoIterator<Item = String>,
        I2: IntoIterator<Item = Vec<f32>>,
    {
        let schema = self.default_table.schema().await?;
        let key_array = StringArray::from_iter_values(filenames);
        let vectors_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vectors.into_iter().map(|v| Some(v.into_iter().map(Some))),
            vec_dim,
        );
        let batches = vec![
            Ok(RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(key_array), Arc::new(vectors_array)],
            )?)
            .map_err(|e| ArrowError::from_external_error(e.into())),
        ];
        let batch_iterator = RecordBatchIterator::new(batches, schema);
        // Create a RecordBatch stream.
        let boxed_batches = Box::new(batch_iterator);

        // add them to the table
        //self.default_table.add(boxed_batches).execute().await?;

        let mut merge_insert = self.default_table.merge_insert(&["filename"]);
        merge_insert
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge_insert.execute(Box::new(boxed_batches)).await?;

        Ok(())
    }

    pub fn sanitize_table_name(name: &str) -> String {
        name.replace("/", "_").replace(":", "_")
    }
}
