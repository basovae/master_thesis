import sys
import time


def format_timespan(seconds: float) -> str:
    '''Format a timespan in seconds to a human-readable string.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Formatted time string (e.g., "1m 30s" or "45s").
    '''
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f'{minutes}m {secs:.1f}s'


def progress_bar(
    iterable,
    length=None,
    bar_size=30,
    prefix='',
    suffix='',
    fill='█',
    print_end='\n',
):
    '''A custom progress bar function that overwrites itself in the terminal.

    Args:
        iterable (iterable): The iterable to loop over.
        length (int, optional): Length of the progress bar (in characters).
            Defaults to None (i.e. will be inferred from iterable).
        bar_size (int, optional): Length of the displayed bar. Defaults to 30 elements.
        prefix (str, optional): Prefix string to display before the progress bar.
        suffix (str, optional): Suffix string to display after the progress bar.
        fill (str, optional): Character to fill the progress bar with. Defaults to '█'.
        print_end (str, optional): End character (e.g., '\r' to overwrite,
            '\n' for new line). Defaults to '\n'.

    Yields:
        The elements from the provided iterable, one at a time.
    '''
    if length is None:
        length = len(iterable)

    def print_bar(progress, elapsed_time):
        percent = 100 * (progress / float(length))
        filled_length = int(bar_size * progress // length)
        bar = fill * filled_length + '-' * (bar_size - filled_length)
        sys.stdout.flush()
        sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix} [Elapsed: {format_timespan(elapsed_time)}]')

    print_bar(0, 0)

    start_time = time.time()

    for i, item in enumerate(iterable, 1):
        yield item
        elapsed = time.time() - start_time
        print_bar(i, elapsed)

    sys.stdout.write(print_end)
