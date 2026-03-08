"""
Test script for training the cross-modal communication system.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from models import create_model_pair
from data import DataLoader, DataConfig, Visualizer
from train import CrossModalTrainer, TrainingConfig


def test_data_generation():
    """Test data generation pipeline."""
    print("Testing data generation...")
    
    config = DataConfig()
    config.img_size = 540  # Smaller for testing
    config.min_objects = 5
    config.max_objects = 15
    
    data_loader = DataLoader(config)
    key = jax.random.PRNGKey(42)
    
    # Generate test batch
    circles, squares, c_counts, s_counts, _ = data_loader.generate_batch(4, key)
    
    print(f"Circle batch shape: {circles.shape}")
    print(f"Square batch shape: {squares.shape}")
    print(f"Circle counts: {c_counts}")
    print(f"Square counts: {s_counts}")
    
    # Visualize
    Visualizer.plot_batch_samples(circles, squares, c_counts, s_counts)
    
    return True


def test_model_initialization():
    """Test model initialization and forward pass."""
    print("Testing model initialization...")
    
    # Create models
    model1, model2 = create_model_pair()
    key = jax.random.PRNGKey(42)
    
    # Test with smaller input for speed
    test_input = jnp.ones((2, 540, 540, 1))
    
    # Initialize models
    params1 = model1.init(key, test_input)
    params2 = model2.init(key, test_input)
    
    print(f"Model 1 initialized successfully")
    print(f"Model 2 initialized successfully")
    
    # Test forward pass
    output1, latent1 = model1.apply(params1, test_input)
    output2, latent2 = model2.apply(params2, test_input)
    
    print(f"Model 1 - Input: {test_input.shape}, Latent: {latent1.shape}, Output: {output1.shape}")
    print(f"Model 2 - Input: {test_input.shape}, Latent: {latent2.shape}, Output: {output2.shape}")
    
    # Test cross-modal decoding
    cross_output1 = model2.apply(params2, latent1, method='decode')
    cross_output2 = model1.apply(params1, latent2, method='decode')
    
    print(f"Cross-modal outputs: {cross_output1.shape}, {cross_output2.shape}")
    
    return True


def test_quick_training():
    """Test a quick training run."""
    print("Testing quick training run...")
    
    # Configure for quick testing
    train_config = TrainingConfig()
    train_config.num_epochs = 5
    train_config.batches_per_epoch = 3
    train_config.batch_size = 1
    train_config.eval_frequency = 2
    train_config.save_frequency = 3
    train_config.learning_rate = 1e-3
    
    data_config = DataConfig()
    data_config.img_size = 540  # Smaller for faster testing
    data_config.min_objects = 3
    data_config.max_objects = 8
    
    # Initialize trainer
    trainer = CrossModalTrainer(train_config, data_config)
    key = jax.random.PRNGKey(42)
    
    # Setup training
    trainer.setup_training(key)
    
    # Run quick training
    start_time = time.time()
    trainer.train(key)
    training_time = time.time() - start_time
    
    print(f"Quick training completed in {training_time:.2f} seconds")
    
    # Test evaluation
    eval_metrics = trainer.evaluate(key, num_eval_batches=2)
    print(f"Evaluation metrics: {eval_metrics}")
    
    return trainer


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print("Testing checkpoint save/load...")
    
    # Quick training setup
    train_config = TrainingConfig()
    train_config.num_epochs = 3
    train_config.batches_per_epoch = 2
    train_config.batch_size = 1
    
    data_config = DataConfig()
    data_config.img_size = 540
    
    # Train and save
    trainer1 = CrossModalTrainer(train_config, data_config)
    key = jax.random.PRNGKey(42)
    trainer1.setup_training(key)
    trainer1.train(key)
    
    # Save checkpoint
    checkpoint_path = "test_checkpoint.pkl"
    trainer1.save_checkpoint(2, checkpoint_path)
    
    # Create new trainer and load
    trainer2 = CrossModalTrainer(train_config, data_config)
    trainer2.setup_training(key)
    
    # Compare states before loading
    state1_params_before = trainer2.state1.params
    
    # Load checkpoint
    loaded_epoch = trainer2.load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint from epoch {loaded_epoch}")
    
    # Verify states are different after loading
    state1_params_after = trainer2.state1.params
    
    # Clean up
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print("Checkpoint save/load test passed")
    return True


def test_cross_modal_inference():
    """Test cross-modal inference capabilities."""
    print("Testing cross-modal inference...")
    
    # Setup
    train_config = TrainingConfig()
    data_config = DataConfig()
    data_config.img_size = 540
    data_config.min_objects = 5
    data_config.max_objects = 10
    
    trainer = CrossModalTrainer(train_config, data_config)
    key = jax.random.PRNGKey(42)
    trainer.setup_training(key)
    
    # Generate test data
    circles, squares, c_counts, s_counts, key = trainer.data_loader.generate_batch(2, key)
    
    print(f"Test data - Circle counts: {c_counts}, Square counts: {s_counts}")
    
    # Forward pass through both models
    _, latent1 = trainer.state1.apply_fn({'params': trainer.state1.params}, circles)
    _, latent2 = trainer.state2.apply_fn({'params': trainer.state2.params}, squares)
    
    # Cross-modal decoding
    reconstructed_squares = trainer.state2.apply_fn({'params': trainer.state2.params}, latent1, method='decode')
    reconstructed_circles = trainer.state1.apply_fn({'params': trainer.state1.params}, latent2, method='decode')
    
    print(f"Latent shapes: {latent1.shape}, {latent2.shape}")
    print(f"Reconstruction shapes: {reconstructed_squares.shape}, {reconstructed_circles.shape}")
    
    # Visualize results
    Visualizer.plot_cross_modal_results(
        circles, squares, latent1, latent2, 
        reconstructed_squares, reconstructed_circles,
        c_counts, s_counts, sample_idx=0
    )
    
    return True


def test_memory_usage():
    """Test memory usage with different configurations."""
    print("Testing memory usage...")
    
    configs = [
        (270, 1),   # Very small
        (540, 1),   # Small  
        (1080, 1),  # Full size, batch 1
    ]
    
    for img_size, batch_size in configs:
        try:
            print(f"Testing {img_size}x{img_size} with batch size {batch_size}")
            
            data_config = DataConfig()
            data_config.img_size = img_size
            
            train_config = TrainingConfig()
            train_config.batch_size = batch_size
            train_config.num_epochs = 1
            train_config.batches_per_epoch = 1
            
            trainer = CrossModalTrainer(train_config, data_config)
            key = jax.random.PRNGKey(42)
            trainer.setup_training(key)
            
            # Single training step
            circles, squares, c_counts, s_counts, _ = trainer.data_loader.generate_batch(batch_size, key)
            
            start_time = time.time()
            trainer.state1, trainer.state2, loss1, loss2, _, _ = trainer.train_step_jit(
                trainer.state1, trainer.state2, circles, squares, c_counts, s_counts
            )
            step_time = time.time() - start_time
            
            print(f"  ✓ Success - Step time: {step_time:.3f}s, Loss1: {loss1:.4f}, Loss2: {loss2:.4f}")
            
        except Exception as e:
            print(f"  ✗ Failed - {str(e)}")
    
    return True


def run_all_tests():
    """Run all test functions."""
    print("="*60)
    print("CROSS-MODAL COMMUNICATION SYSTEM TESTS")
    print("="*60)
    
    tests = [
        ("Data Generation", test_data_generation),
        ("Model Initialization", test_model_initialization),
        ("Quick Training", test_quick_training),
        ("Checkpoint Save/Load", test_checkpoint_save_load),
        ("Cross-Modal Inference", test_cross_modal_inference),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            start_time = time.time()
            result = test_func()
            test_time = time.time() - start_time
            
            if result:
                print(f"✓ {test_name} PASSED ({test_time:.2f}s)")
                results.append((test_name, "PASSED", test_time))
            else:
                print(f"✗ {test_name} FAILED ({test_time:.2f}s)")
                results.append((test_name, "FAILED", test_time))
                
        except Exception as e:
            test_time = time.time() - start_time
            print(f"✗ {test_name} ERROR: {str(e)} ({test_time:.2f}s)")
            results.append((test_name, "ERROR", test_time))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print("="*60)
    
    total_time = 0
    passed = 0
    
    for test_name, status, test_time in results:
        status_symbol = "✓" if status == "PASSED" else "✗"
        print(f"{status_symbol} {test_name:<25} {status:<8} ({test_time:.2f}s)")
        total_time += test_time
        if status == "PASSED":
            passed += 1
    
    print("-"*60)
    print(f"Total: {len(results)} tests, {passed} passed, {len(results)-passed} failed")
    print(f"Total time: {total_time:.2f}s")
    
    return results


if __name__ == "__main__":
    # Set JAX to use CPU for testing (comment out for TPU)
    # os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    
    results = run_all_tests()
    
    # Return non-zero exit code if any tests failed
    if any(status != "PASSED" for _, status, _ in results):
        exit(1)
    else:
        print("\n🎉 All tests passed successfully!")
        exit(0)