#!/usr/bin/env python3
"""
Script to download all configured models for Local Model Manager
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_model_manager.core.model_manager import ModelDownloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Download all models"""
    config_path = Path(__file__).parent.parent / "configs" / "model_configs.yaml"
    downloader = ModelDownloader(str(config_path))

    logger.info("Starting model downloads...")

    results = await downloader.download_all_models()

    logger.info("Download Results:")
    for model_id, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"  {model_id}: {status}")

    # Verify downloads
    logger.info("\nVerifying downloads...")
    for model_id in results.keys():
        is_valid = await downloader.verify_model(model_id)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        logger.info(f"  {model_id}: {status}")

    available_models = downloader.list_available_models()
    logger.info(f"\nAvailable models: {len(available_models)}")
    for model_id in available_models:
        logger.info(f"  - {model_id}")

if __name__ == "__main__":
    asyncio.run(main())