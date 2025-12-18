import pandas as pd
import torch

class RLDataLoader:
    '''Loads custom PyTorch DataLoaders for Reinforcement Learning (RL) applications.

    Args:
        data_train (pd.DataFrame): DataFrame with train dataset in standard
            tabular format.
        data_test (pd.DataFrame): DataFrame with validation dataset.
        shuffle (bool, optional): set to True to have the data reshuffled at
            every epoch. Defaults to False.

    Returns:
        tuple: A tuple containing two DataLoader objects:
            - train_loader (DataLoader): DataLoader for the training dataset with
                pairs of sequential states of length `window_size`.
            - val_loader (DataLoader): DataLoader for the val dataset, which
                includes the last `window_size` days of training data
                concatenated with the val data.

    Notes:
        - The training dataset contains sequences derived solely from the
            training data.
        - The val dataset includes the last `window_size` days of training data
            to ensure continuity for the first sample.
    '''
    def __init__(
        self,
        data_train: pd.DataFrame,
        data_val: pd.DataFrame,
        shuffle: bool = False,
    ):
        self.data_train = data_train
        self.data_val = data_val
        self.shuffle = shuffle

    class RLDataset(torch.utils.data.Dataset):
        '''Custom PyTorch Dataset for Reinforcement Learning (RL) applications.

        This dataset generates samples consisting of pairs of sequential states: 
        - The `current state` represented by a window of historical data.
        - The `next state` represented by the subsequent window of data. 

        Each entry in the dataset corresponds to two consecutive windows of size `window_size`. 

        Args:
            df (pd.DataFrame): Input DataFrame with each column being a feature
                and each row being an instance or timepoint.
            window_size (int): Number of consecutive data points to consider for each state window.
        '''
        def __init__(self, df: pd.DataFrame, window_size: int, forecast_size: int):
            self.data = df.values
            self.window_size = window_size
            self.forecast_size = forecast_size

        def __len__(self) -> int:
            # -1 to skip last window which isn't full length
            return len(self.data) // (self.window_size + self.forecast_size) - 1

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            train_start = idx*self.window_size
            train_end = train_start + self.window_size
            test_start = train_end
            test_end = train_end + self.window_size + self.forecast_size
            
            input_data = self.data[train_start:train_end]
            if self.forecast_size > 0:
                input_data = arima_forecast(input_data, self.forecast_size)
            input_window = torch.tensor(
                input_data,
                dtype=torch.float32,
            )
            target_window = torch.tensor(
                self.data[test_start:test_end],
                dtype=torch.float32,
            )
            return input_window, target_window

    def __call__(
        self,
        batch_size: int,
        window_size: int,
        forecast_size: int = 0,
    ):
        '''Returns DataLoader objects for the training and validation datasets.
        
        Args:
            batch_size (int): Number of samples per batch.
            window_size (int): Number of consecutive data points to consider for
                each state window.
            forecast_size (int, optional): Number of days to forecast using ARIMA.
                Defaults to 0.
        '''
        train_dataset = self.RLDataset(
            self.data_train,
            window_size=window_size,
            forecast_size=forecast_size,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=self.shuffle,
        )
        
        val_data = pd.concat((
            self.data_train[-window_size:],
            self.data_val
        )) # add last window from the train data to ensure continuity
        val_dataset = self.RLDataset(
            val_data,
            window_size=window_size,
            forecast_size=forecast_size,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=self.shuffle,
        )
        
        return train_loader, val_loader
    
    @property
    def number_of_assets(self):
        '''Returns the number of financial assets in the dataset.'''
        return self.data_train.shape[1]