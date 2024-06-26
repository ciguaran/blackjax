# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Progress bar decorators for use with step functions.
Adapted from Jeremie Coullon's blog post :cite:p:`progress_bar`.
"""
from fastprogress.fastprogress import progress_bar
from jax import lax
from jax.experimental import io_callback


def progress_bar_scan(num_samples, print_rate=None):
    "Progress bar for a JAX scan"
    progress_bars = {}

    if print_rate is None:
        if num_samples > 20:
            print_rate = int(num_samples / 20)
        else:
            print_rate = 1  # if you run the sampler for less than 20 iterations

    def _define_bar(arg):
        del arg
        progress_bars[0] = progress_bar(range(num_samples))
        progress_bars[0].update(0)

    def _update_bar(arg):
        progress_bars[0].update_bar(arg + 1)

    def _close_bar(arg):
        del arg
        progress_bars[0].on_iter_end()

    def _update_progress_bar(iter_num):
        "Updates progress bar of a JAX scan or loop"
        _ = lax.cond(
            iter_num == 0,
            lambda _: io_callback(_define_bar, None, iter_num),
            lambda _: None,
            operand=None,
        )

        _ = lax.cond(
            # update every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) | (iter_num == (num_samples - 1)),
            lambda _: io_callback(_update_bar, None, iter_num),
            lambda _: None,
            operand=None,
        )

        _ = lax.cond(
            iter_num == num_samples - 1,
            lambda _: io_callback(_close_bar, None, None),
            lambda _: None,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            _update_progress_bar(iter_num)
            return func(carry, x)

        return wrapper_progress_bar

    return _progress_bar_scan
