def run_in_executor(func, *args, **kwargs):
    """
    Run a function in a separate thread using ThreadPoolExecutor.
    
    Args:
        func (callable): The function to run.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
    
    Returns:
        Future: A Future object representing the execution of the function.
    """
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
    return future