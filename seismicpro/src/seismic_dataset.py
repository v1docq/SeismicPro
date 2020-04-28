"""File contains seismic dataset."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tdigest import TDigest

from ..batchflow import Dataset
from .seismic_index import FieldIndex
from .seismic_batch import SeismicBatch
from .utils import check_unique_fieldrecord_across_surveys
from .file_utils import load_horizon


class SeismicDataset(Dataset):
    """Dataset for seismic data."""

    def __init__(self, index, batch_class=SeismicBatch, preloaded=None, *args, **kwargs):
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, *args, **kwargs)
        self.horizon = None

    def find_sdc_params(self, component, speed, loss, indices=None, time=None, initial_point=None,
                        method='Powell', bounds=None, tslice=None, **kwargs):
        """ Finding an optimal parameters for correction of spherical divergence.

        Parameters
        ----------
        component : str
            Component with shot gathers.
        speed : array
            Wave propagation speed depending on the depth.
            Speed is measured in milliseconds.
        loss : callable
            Function to minimize.
        indices : array-like, optonal
            Which items from dataset to use in parameter estimation.
            If `None`, defaults to first element of dataset.
        time : array, optional
           Trace time values. If `None` defaults to self.meta[src]['samples'].
           Time measured in either in samples or in milliseconds.
        initial_point : array of 2
            Started values for $v_{pow}$ and $t_{pow}$.
            If None defaults to $v_{pow}=2$ and $t_{pow}=1$.
        method : str, optional, default ```Powell```
            Minimization method, see ```scipy.optimize.minimize```.
        bounds : sequence, optional
            Sequence of (min, max) optimization bounds for each parameter.
            If `None` defaults to ((0, 5), (0, 5)).
        tslice : slice, optional
            Lenght of loaded traces.

        Returns
        -------
            : array
            Coefficients for speed and time.

        Raises
        ------
        ValueError : If Index is not FieldIndex.

        Note
        ----
        To save parameters as SeismicDataset attribute use ```save_to=D('attr_name')``` (works only
        in pipeline).
        If you want to save parameters to pipeline variable use save_to argument with following
        syntax: ```save_to=V('variable_name')```.
        """
        if not isinstance(self.index, FieldIndex):
            raise ValueError("Index must be FieldIndex, not {}".format(type(self.index)))

        if indices is None:
            indices = self.indices[:1]

        batch = self.create_batch(indices).load(components=component, fmt='segy', tslice=tslice)
        field = getattr(batch, component)[0]
        samples = batch.meta[component]['samples']

        bounds = ((0, 5), (0, 5)) if bounds is None else bounds
        initial_point = (2, 1) if initial_point is None else initial_point

        time = samples if time is None else np.array(time, dtype=int)
        step = np.diff(time[:2])[0].astype(int)
        speed = np.array(speed, dtype=int)[::step]
        args = field, time, speed

        func = minimize(loss, initial_point, args=args, method=method, bounds=bounds, **kwargs)
        return func.x

    def find_equalization_params(self, batch, component, survey_id_col, sample_size=10000,
                                 container_name='equal_params', **kwargs):
        """ Estimates 95th percentile of absolute values for each seismic survey
        in dataset for equalization.

        This method utilizes t-digest structure for batch-wise estimation of rank-based statistics,
        namely 95th percentile.

        Parameters
        ----------
        batch : SeismicBatch or B() named expression.
            Current batch from pipeline.
        component : str
            Component with shot gathers.
        survey_id_col : str
            Column in index that indicate names of seismic
            surveys from different seasons.
        sample_size: int, optional
            Number of elements to draw from each shot gather to update
            estimates if TDigest. Time for each update grows linearly
            with `sample_size`. Default is 10000.
        container_name: str, optional
            Name of the `SeismicDataset` attribute to store a dict
            with estimated percentile. Also contains `survey_id_col`
            key and corresponding value.
        kwargs: misc
            Parameters for TDigest objects.

        Raises
        ------
        ValueError : If index is not FieldIndex.
        ValueError : If shot gather with same id is contained in more
                     than one survey.

        Note
        ----
        Dictoinary with estimated percentile can be obtained from pipeline using `D(container_name)`.
        """
        if not isinstance(self.index, FieldIndex):
            raise ValueError("Index must be FieldIndex, not {}".format(type(self.index)))

        private_name = '_' + container_name
        params = getattr(self, private_name, None)
        if params is None:
            surveys = np.unique(self.index.get_df()[survey_id_col])
            delta, k = kwargs.pop('delta', 0.01), kwargs.pop('K', 25)
            params = dict(zip(surveys, [TDigest(delta, k) for _ in surveys]))
            setattr(self, private_name, params)

        for idx in batch.indices:
            surveys_by_fieldrecord = np.unique(batch.index.get_df(index=idx)[survey_id_col])
            check_unique_fieldrecord_across_surveys(surveys_by_fieldrecord, idx)
            survey = surveys_by_fieldrecord[0]

            pos = batch.get_pos(None, component, idx)
            sample = np.random.choice(getattr(batch, component)[pos].reshape(-1), size=sample_size)

            params[survey].batch_update(np.absolute(sample))

        statistics = dict([survey, digest.percentile(95)]
                          for survey, digest in params.items() if digest.n > 0)
        statistics['survey_id_col'] = survey_id_col
        setattr(self, container_name, statistics)

    def find_avo_distribution(self, batch, component, class_size, window=None, field_type='same', field_ratio=10,
                              horizon=None, horizon_width=10, container_name='avo_classes'):
        """Calculate AVO distribution for all dataset and save result to `container_name`.

        Parameters
        ----------
        batch : SeismicBatch or B() named expression.
            Current batch from pipeline.
        component : str
            Component with shot gathers.
        class_size : int or array-like
            Lenght of one class or lenght of each class if iterable.
        window : array-like with size 2 or string
            If array-like, it is an interval in ms where to
            constract AVO distribution in the following order [from ms, to ms].
            Else, path to horizon.
        field_type : 'same' or 'diff', optional, default: 'same'
            If all seismograms have the same lenght and same offset use 'same', otherwise use 'diff'.
        field_ratio : int, default 10
            Used only with `field_type` 'diff'. Should be bigger
            then ratio between longest and shortest seismograms.
        horizon : str, optional
            Path to horizon. If not None, AVO will be calculated based on this
            data even if `window` is specified.
        horizon_width : int, default 10
            Width of the window along the horizon.
        contaier_name : str, optional
            Name of the `AvoDataset` attribute to store a dict
            with estimated percentile.

        Raises
        ------
        ValueError : If trases is not sorted by `offset`.
        ValueError : If `window` is not array-like.
        ValueError : If `field_type` is not 'same' or 'diff'.

        Note
        ----
        1. This function works properly only with CDP index name with sorting by `offset`.
        2. Dictoinary with estimated AVO classes can be obtained from pipeline using `D(container_name)`.
        3. Horizon parameter have a priority over window.
        """
        sorting = batch.meta[component]['sorting']
        if sorting != 'offset':
            ValueError("Wrong sorting type. Should be 'offset' but given {}.".format(sorting))

        if horizon is not None:
            if self.horizon is None:
                col_names = batch.index.get_df().columns
                for title in ['offset', 'CROSSLINE_3D', 'INLINE_3D']:
                    if not title in col_names:
                        raise ValueError("Missing extra_header: {0} in Index dataframe. Add {0} as `extra_headers`"
                                         "to Index. Your list of headers is the following {1}".format(title,
                                                                                                      col_names))
                if isinstance(horizon, str):
                    self.horizon = load_horizon(horizon)
                elif isinstance(horizon, (list, tuple, np.ndarray)):
                    self.horizon = pd.DataFrame(horizon, columns=['INLINE_3D', 'CROSSLINE_3D', 'time'])
                else:
                    raise ValueError("Wrong type of `horizon` variable. Should be array-like or str"
                                     ", not {}".format(type(horizon)))
        elif window is not None:
            if not isinstance(window, (list, tuple, np.ndarray)):
                raise ValueError("Wrong type of `window` variable. Should be array-like with size 2"
                                 ", not {}".format(type(window)))
            window = np.round(np.array(window)/batch.meta[component]['samples'][1]).astype(np.int32)
        else:
            raise ValueError("One of variables `horizon` or `window` should be determined.")

        params = getattr(self, container_name, None)

        class_size = np.array(class_size) if isinstance(class_size, (tuple, list)) else class_size
        if params is not None:
            storage_size = params.shape[1]
        else:
            field = getattr(batch, component)[0]
            if field_type == 'same':
                storage_size = np.ceil(np.max(batch.index.get_df()['offset']) / class_size).astype(int)
            elif field_type == 'diff':
                storage_size = field.shape[0] * field_ratio
            else:
                raise ValueError("`field_type` should be 'same' or 'diff', not {}".format(field_type))

        params = self._update_avo_params(params, batch, component, class_size, storage_size,
                                         window, horizon_width, self.horizon)
        setattr(self, container_name, params)

    def _update_avo_params(self, params, batch, component, class_size, storage_size, window, horizon_width, horizon):
        """One step of AVO. """
        for idx in batch.indices:
            pos = batch.get_pos(None, component, idx)
            field = getattr(batch, component)[pos]

            batch_df = batch.index.get_df(index=idx)
            offset = np.sort(batch_df['offset'])
            if horizon is not None:
                crossline, inline = batch_df[['CROSSLINE_3D', 'INLINE_3D']].iloc[0]
                window = horizon[(horizon['CROSSLINE_3D'] == crossline) & \
                                 (horizon['INLINE_3D'] == inline)]['time'].iloc[0]
                window = np.array([window, window+horizon_width]/batch.meta[component]['samples'][1]).astype(np.int32)
            if isinstance(class_size, int):
                step_list = np.arange(0, offset[-1]+class_size, class_size)
            else:
                step_list = np.append(step_list[step_list < offset[-1]], offset[-1]+1)

            storage = np.zeros(storage_size)
            for i, stp in enumerate(step_list[:-1]):
                subfield = field[np.array(offset > stp) & np.array(offset < step_list[i+1])]
                if subfield.size == 0:
                    storage[i] = 0
                    continue
                storage[i] = np.mean(np.mean(subfield[:, window[0]: window[1]+1]**2, axis=1)**.5)
            params = np.array([storage]) if params is None else np.append(params, [storage], axis=0)
        return params
