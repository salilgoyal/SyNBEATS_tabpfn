import torch
import pandas as pd
from darts import TimeSeries, concatenate
from darts.models import NBEATSModel
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import logging
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import norm
import warnings
# from pytorch_lightning.utilities.exceptions import PossibleUserWarning

warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

class SyNBEATS:
    
    # self.dta: Pandas DataFrame with colnames = ['id', 'time', 'Y_obs']
    # treat_ids: List of Integers
    # target_time: Integer
    def __init__(self, dta, treat_ids, target_time, control_ids=None,
        date_format=None,
        input_size=1, output_size=1,
        ):
        
        assert len(dta['time'])>1, "Time span has to be at least 2 unit time long"
        
        self.dta = dta
        self.treat_ids = treat_ids
        
        self.date_format = date_format
        if self.date_format:
            target_time = pd.to_datetime(target_time, format=self.date_format)
            self.dta["time"] = pd.to_datetime(self.dta["time"], format=self.date_format)
                
        self._step_size = self.dta['time'][1]-self.dta['time'][0]
        # print(self._step_size)
        if control_ids:
            self.control_ids = control_ids
        else:
            self.control_ids = list(set(self.dta['id']) - set(self.treat_ids))
            
        self.target_time = target_time
        self.input_size = input_size
        self.output_size = output_size
                
        self._prepare_data()

    
    def _prepare_data(self):            

        self.dta["id_tr"] = self.dta["id"].apply(lambda x: 1 if x in self.treat_ids else 0)
        self.dta["tr"] = np.where((self.dta["id"].isin(self.treat_ids)) & (self.dta["time"] >= self.target_time), 1, 0)

        # cov_list_all = [TimeSeries.from_dataframe(self.dta[self.dta['id'] == cid], 'time', 'Y_obs').astype(np.float32) for cid in self.control_ids]
        cov_list_all = [TimeSeries.from_dataframe(self.dta[self.dta['id'] == cid], 'time', 'Y_obs') for cid in self.control_ids]
        cov_list_all = concatenate(cov_list_all, axis=1)
        
        cov_train_list = [TimeSeries.from_dataframe(self.dta[(self.dta['id'] == c_id) & (self.dta['time'] < self.target_time)],
                                                'time', 'Y_obs') for c_id in self.control_ids]
        cov_list_train = concatenate(cov_train_list, axis=1)

        ts_all_list = [TimeSeries.from_dataframe(self.dta[self.dta['id'] == treat_id], 'time', 'Y_obs') for treat_id in self.treat_ids]
        ts_list_all = concatenate(ts_all_list, axis=1)
        
        
        
        ts_train_list = [TimeSeries.from_dataframe(self.dta[(self.dta['id'] == treat_id) & (self.dta['time'] < self.target_time)],
                                                'time', 'Y_obs') for treat_id in self.treat_ids]
        ts_list_train = concatenate(ts_train_list, axis=1)

        ts_test_list = [TimeSeries.from_dataframe(self.dta[(self.dta['id'] == treat_id) & (self.dta['time'] >= self.target_time)],
                                                'time', 'Y_obs') for treat_id in self.treat_ids]
        ts_list_test = concatenate(ts_test_list, axis=1)

        
        self.ts_list_all = ts_list_all
        self.ts_list_test = ts_list_test
        self.ts_list_train = ts_list_train
        self.cov_list_all = cov_list_all
        self.cov_list_train = cov_list_train
        

    def train(self, epochs=1500, lr=1e-4, batch_size=1024,
        patience=20, min_delta=0.005,
        use_gpu=None,
        verbose=True):

        early_stopping = EarlyStopping(
            monitor="train_loss", 
            patience=patience, 
            min_delta=min_delta
        )
        
        pl_trainer_kwargs = {"callbacks": [early_stopping], "accelerator": "gpu" if use_gpu else "cpu"}
        if use_gpu:
            pl_trainer_kwargs["devices"] = use_gpu

        the_model = NBEATSModel(
            input_chunk_length=self.input_size, 
            output_chunk_length=self.output_size, 
            n_epochs=epochs, 
            batch_size=batch_size, 
            optimizer_kwargs={'lr': lr}, 
            pl_trainer_kwargs=pl_trainer_kwargs
        )

        the_model.fit(series=self.ts_list_train, past_covariates=self.cov_list_all, verbose=verbose) 
        
        self.model = the_model
        backtest = self.model.historical_forecasts(series=self.ts_list_train, past_covariates=self.cov_list_train, retrain=False)
        self.backtest = backtest
    
    def predict(self, pred_length=-1, df=False, verbose=True):
        if pred_length == -1:
            pred_length = len(self.dta[(self.dta['time'] >= self.target_time) & (self.dta['id'] == self.treat_ids[0])])
            
        darts_pred = self.model.predict(n=pred_length, series=self.ts_list_train, past_covariates=self.cov_list_all, verbose=verbose)

        if df:
            return darts_pred.pd_dataframe()
        return darts_pred
    
    def plot_predictions(self, darts_pred, predict_pretreatment=False, title=None, l_obs='Observed', l_pred='Predicted'):
        plt.figure()
        if not predict_pretreatment:
            if len(self.treat_ids) == 1:
                self.ts_list_train.append(darts_pred).plot(label=l_pred, color='blue')
                self.ts_list_all.plot(label=l_obs)
            else:
                self.ts_list_train.append(darts_pred).mean(1).plot(label=l_pred, color='blue')
                self.ts_list_all.mean(1).plot(label=l_obs)

            plt.axvline(x=self.target_time-self._step_size, color='gray', linestyle='--', label='Last Treated Time')
            
            if title:
                plt.title(title)
            else:
                plt.title('Prediction Plot without Pre-Treatment Predictions')
            plt.legend()
            plt.savefig('predictions_without_pre.png')

            plt.show()
        else:
            if len(self.treat_ids) == 1:
                predicted = self.backtest.prepend(self.ts_list_all[0]).append(darts_pred)
                
                predicted_values = predicted.values()  # Extracting values
                predicted_times = predicted.time_index  # Extracting time index

                plt.plot(predicted_times, predicted_values, label=l_pred, alpha=0.6, linewidth=2, color='b')

                ts_all_values = self.ts_list_all.values()  # Extracting values
                ts_all_times = self.ts_list_all.time_index  # Extracting time index

                plt.plot(ts_all_times, ts_all_values, label=l_obs, alpha=0.4, linewidth=2, color='black')

            else:
                predicted = self.backtest.prepend(self.ts_list_all[0]).append(darts_pred).mean(1)
                
                predicted_values = predicted.values()  # Extracting values
                predicted_times = predicted.time_index  # Extracting time index

                plt.plot(predicted_times, predicted_values, label=l_pred, alpha=0.6, linewidth=2, color='b')

                mean_ts_all_values = self.ts_list_all.mean(1).values()  # Extracting mean values
                mean_ts_all_times = self.ts_list_all.mean(1).time_index  # Extracting time index of mean values

                plt.plot(mean_ts_all_times, mean_ts_all_values, label=l_obs, alpha=0.4, linewidth=2, color='black')

            plt.axvline(x=self.target_time-self._step_size, color='gray', linestyle='--', label='Last Treated Time')
            if title:
                plt.title(title)
            else:
                plt.title('Prediction Plot with Pre-Treatment Predictions')
            plt.legend()
            plt.savefig('predictions_with_pre.png')
            plt.show()

    
    def backtest(self, df=False, retrain=True):
        
        backtest = self.model.historical_forecasts(series=ts_list_all, past_covariates=self.cov_list_all, retrain=retrain)

        if df:
            return backtest.pd_dataframe().reset_index()
        return backtest
        


    def plot_backtest(self, backtest, title="Backcast Plot", l_obs='Observed', l_pred='Predicted'):
        self.ts_list_all.plot(label=l_obs)
        backtest.plot(label=l_pred)
        
        plt.title(title)
        plt.legend()
        plt.show()
        
    def _gap(self):
        gap = self.ts_list_all - self.backtest.prepend(self.ts_list_all[0]).append(self.predict(verbose=False))
        if len(self.treat_ids) == 1:
            return gap
        else:
            return gap.mean(1)

    def plot_gap(self, l='Observed - Predicted', title="Gap Plot"):
        gap = self._gap()
        gap.plot(label=l)

        plt.axhline(y=0, color='gray', linestyle='--')
        lim = max(abs(gap.values().flatten()))*1.05
        plt.ylim(-lim, lim)
        
        plt.axvline(x=self.target_time-self._step_size, color='gray', linestyle='--', label='Last Treated Time')
        
        plt.title(title)
        plt.legend()
        plt.savefig('gap.png')
        plt.show()


    def average_treatment_effect(self):
        gap = self._gap()
        ate = gap.values().mean().item()
        return ate

    def std_treatment_effect(self):
        gap = self._gap()
        std = gap.values().std().item()
        return std
    

    def placebo_test(self, control_ids=None, use_gpu=None, plot=True):
        if not control_ids:
            control_ids = self.control_ids
            
        if len(self.treat_ids) != 1:
            raise Exception('Placebo test for multiple treated units is not supported, please see documentation for more information')
       

        placebo_effects = {}

        print("Starting the placebo test...")

        pbar = tqdm(control_ids, desc='Processing placebo for control id')
        ates = []
        gaps = []
        for cid in pbar:
            pbar.set_description(f'Processing placebo for control id {cid}')
            placebo_model = SyNBEATS(self.dta, [cid], self.target_time, list(set(self.control_ids) - set([cid])), None,
                                    self.input_size, self.output_size)
            placebo_model.train(verbose=False, use_gpu=use_gpu)
            placebo_effects[cid] = placebo_model.predict(verbose=False)
            ates.append(placebo_model.average_treatment_effect())
            gaps.append(placebo_model._gap())
            del placebo_model
            torch.cuda.empty_cache()
        print("Placebo test completed.")
        
        if plot:
            plt.figure()
            
            lim = 0
            plotted = False
            for gap in gaps:
                if not plotted:
                    plt.plot(gap.time_index.tolist(), gap.values().squeeze(), color='gray', linewidth=0.5, label='Controls')
                    plotted = True
                else:
                    plt.plot(gap.time_index.tolist(), gap.values().squeeze(), color='gray', linewidth=0.5)
                lim = max(lim, max(abs(gap.values().squeeze()))*1.05)

            gap = self._gap()
            plt.plot(gap.time_index.tolist(), gap.values().squeeze(), color='b', linewidth=2, label='Treated')

            lim = max(lim, max(abs(gap.values().squeeze()))*1.05)
            plt.axhline(y=0, color='gray', linestyle='--')
            plt.ylim(-lim, lim)

            plt.axvline(x=self.target_time-self._step_size, color='gray', linestyle='--', label='Last Treated Time')

            plt.title('Placebo Effect Plot')
            plt.legend()
            plt.savefig('placebo.png')
            plt.show()

        
        actual_ate = self.average_treatment_effect()
        placebo_means = np.mean(ates)
        std_dev = np.std(ates)
        z_score = (actual_ate - placebo_means) / std_dev

        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        return placebo_effects, p_value
    
    def _plot_placebo_for_cid(self, cid, gaps):
        
        
        plt.figure()
            
        lim = 0
        for gap in gaps:
            plt.plot(gap.time_index.tolist(), gap.values().squeeze(), color='gray', linewidth=0.5)
            lim = max(lim, max(abs(gap.values().squeeze()))*1.05)

        gap = self._gap()
        plt.plot(gap.time_index.tolist(), gap.values().squeeze(), color='b', linewidth=2)

        lim = max(lim, max(abs(gap.values().squeeze()))*1.05)
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.ylim(-lim, lim)

        plt.axvline(x=self.target_time-self._step_size, color='gray', linestyle='--', label='Last Treated Time')

        plt.title('Placebo Effect Plot')
        plt.legend(['Controls', 'Treated'])
        # plt.savefig('placebo.png')
        plt.show()
        

    def check_significance(self, placebo_effects, alpha=0.05):
        actual_ate = self.average_treatment_effect()
        placebo_means = np.mean(list(placebo_effects.values()))
        std_dev = np.std(list(placebo_effects.values()))
        z_score = (actual_ate - placebo_means) / std_dev

        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        critical_value = norm.ppf(1 - alpha/2)

        return abs(z_score) > critical_value, p_value

