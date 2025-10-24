from models import FinancialForecastingModel
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import (
    NHiTSModel,
    NBEATSModel,
    TCNModel,
    TransformerModel,
    RegressionModel,
    RNNModel,
    TFTModel,
)
from sklearn.ensemble import RandomForestRegressor


class DartsFinancialForecastingModel(FinancialForecastingModel):
    """A financial forecasting model based on the Darts library."""
    def __init__(self, model_name, data_processor, model_config):
        self.data_processor = data_processor
        self.scaler = None
        self.model_config = model_config
        self.model = self.initalize_model(model_name)

    def initalize_model(self, model_name):
        """
        Creates the model with hyperparameters tuned for noisy, mean-reverting
        ratio/log-return series typical in pairs trading.
        """
        lr = 5e-4
        bs = self.model_config.BATCH_SIZE
        n_epochs = self.model_config.N_EPOCHS
        in_len = self.model_config.INPUT_CHUNK_LENGTH
        out_len = self.model_config.OUTPUT_CHUNK_LENGTH

        trainer = {
            "accelerator": "auto",
            "devices": 1,
            "enable_progress_bar": False,
        }

        if model_name == "nbeats":
            return NBEATSModel(
                input_chunk_length=in_len,
                output_chunk_length=out_len,
                num_layers=4,
                num_blocks=1,
                num_stacks=8,
                layer_widths=256,
                dropout=0.1,
                n_epochs=n_epochs,
                batch_size=bs,
                model_name="nbeats",
                optimizer_kwargs={"lr": lr},
                pl_trainer_kwargs=trainer,
            )

        elif model_name == "nhits":
            return NHiTSModel(
                input_chunk_length=in_len,
                output_chunk_length=out_len,
                num_layers=3,
                num_blocks=1,
                num_stacks=4,
                layer_widths=128,
                dropout=0.1,
                n_epochs=n_epochs,
                batch_size=bs,
                model_name="nhits",
                optimizer_kwargs={"lr": lr},
                pl_trainer_kwargs=trainer,
            )

        elif model_name == "transformer":
            return TransformerModel(
                input_chunk_length=in_len,
                output_chunk_length=out_len,
                n_epochs=n_epochs,
                batch_size=bs,
                nhead=8,
                d_model=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                dim_feedforward=256,
                norm_type="LayerNormNoBias",
                dropout=0.1,
                model_name="transformer",
                optimizer_kwargs={"lr": lr},
                pl_trainer_kwargs=trainer,
            )

        elif model_name == "tcn":
            return TCNModel(
                input_chunk_length=in_len,
                output_chunk_length=out_len,
                kernel_size=3,
                num_filters=32,
                num_layers=6,
                dilation_base=2,
                weight_norm=True,
                dropout=0.1,
                n_epochs=n_epochs,
                batch_size=bs,
                model_name="tcn",
                optimizer_kwargs={"lr": lr},
                pl_trainer_kwargs=trainer,
            )

        elif model_name == "random_forest":
            rf_lags = max(6, min(48, in_len if in_len is not None else 24))
            return RegressionModel(
                lags=rf_lags,
                model=RandomForestRegressor(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            )

        elif model_name == "lstm":
            train_len = (in_len or 50) + (out_len or 1)
            return RNNModel(
                model="LSTM",
                input_chunk_length=in_len,
                output_chunk_length=out_len,
                training_length=train_len,
                hidden_dim=128,
                n_rnn_layers=2,
                dropout=0.2,
                batch_size=bs,
                n_epochs=n_epochs,
                random_state=42,
                optimizer_kwargs={"lr": lr},
                pl_trainer_kwargs=trainer,
            )

        elif model_name == "tft":
            return TFTModel(
                input_chunk_length=in_len,
                output_chunk_length=out_len,
                hidden_size=128,
                lstm_layers=2,
                dropout=0.1,
                batch_size=bs,
                n_epochs=n_epochs,
                add_relative_index=True,
                add_encoders=None,
                random_state=42,
                model_name="tft",
                optimizer_kwargs={"lr": lr},
                pl_trainer_kwargs=trainer,
            )

        else:
            raise ValueError("Invalid model name.")

    def split_and_scale_data(self, train_ratio=0.7, validation_ratio=0.1):
        """
        Splits the data into training, validation, and test sets and applies scaling.
        Keeps your original API; you can pass different ratios when calling if needed.
        """
        series = self.data_processor.get_ratio_time_series()

        num_observations = len(series)
        train_end_index = int(num_observations * train_ratio)
        validation_end_index = int(num_observations * (train_ratio + validation_ratio))

        train_series = series[:train_end_index]
        val_series = series[train_end_index:validation_end_index]
        test_series = series[validation_end_index:]

        self.scaler = Scaler()
        train_series_scaled = self.scaler.fit_transform(train_series)
        valid_series_scaled = self.scaler.transform(val_series)
        test_series_scaled = self.scaler.transform(test_series)

        return train_series_scaled, valid_series_scaled, test_series_scaled

    def train(self, train_series, validation_series):
        """Trains the model."""
        if isinstance(self.model, RegressionModel):
            self.model.fit(train_series, val_series=validation_series)
        else:
            self.model.fit(train_series, val_series=validation_series, verbose=False)

    def predict_future_values(self, test_series):
        """Makes future value predictions based on the test series."""
        return self.model.predict(self.model_config.OUTPUT_CHUNK_LENGTH, series=test_series)

    def generate_predictions(self, test_series):
        """
        Generates predictions for each rolling window of the test series.
        Keeps your original logic and returns values in the original scale.
        """
        transformed_series = []
        for i in range(len(test_series) - self.model_config.INPUT_CHUNK_LENGTH):
            transformed_series.append(test_series[i : i + self.model_config.INPUT_CHUNK_LENGTH])

        pred_series = self.predict_future_values(transformed_series)

        predicted_values = []
        for pred in pred_series:
            predicted_values.append(pred.values()[0][0])

        predicted_df = pd.DataFrame(predicted_values, columns=["predicted"])
        tseries_predicted = TimeSeries.from_dataframe(predicted_df, value_cols="predicted")
        tseries_predicted = self.scaler.inverse_transform(tseries_predicted).to_series(copy=True)
        predicted_values = tseries_predicted.values.tolist()

        return predicted_values

    def get_true_values(self, test_series):
        """Retrieves true values from the test series after scaling back."""
        test_series_inverse = self.scaler.inverse_transform(test_series)
        true_df = test_series_inverse.to_series(copy=True)
        true_values = true_df[self.model_config.INPUT_CHUNK_LENGTH :].values.tolist()
        return true_values
