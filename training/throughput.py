import torch
import time
from typing import Tuple

def test_throughput(data_loader, model, logger) -> float:
    """
    Test the throughput of a model with a given data loader.
    
    Args:
        data_loader: DataLoader providing input batches
        model: Model to test
        logger: Logger for output
        
    Returns:
        Throughput in samples per second
    """
    model.eval()
    
    # Get first batch
    videos, labels = next(iter(data_loader))
    videos = videos.cuda(non_blocking=True)
    batch_size = videos.shape[0]
    
    # Generate text prompts using class labels
    with torch.no_grad():
        classnames = [model.prompt_learner.classnames[label] for label in labels]
        prompts = model.prompt_learner(classnames)
        tokenized_prompts = model.tokenized_prompts
    
    # Warmup
    with torch.no_grad():
        for _ in range(50):
            model(videos, prompts, tokenized_prompts)
        torch.cuda.synchronize()
        
        # Measure throughput
        logger.info("Throughput averaged with 30 times")
        start_time = time.time()
        for _ in range(30):
            model(videos, prompts, tokenized_prompts)
        torch.cuda.synchronize()
        end_time = time.time()
        
    # Calculate throughput
    throughput = 30 * batch_size / (end_time - start_time)
    logger.info(f"Batch size {batch_size} throughput {throughput}")
    
    return throughput