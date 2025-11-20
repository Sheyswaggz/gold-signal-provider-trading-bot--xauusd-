# XAUUSD Trading Signal Bot

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)](https://fastapi.tiangolo.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready trading signal bot for XAUUSD (Gold/USD) that provides real-time technical analysis and trading signals using clean architecture principles.

## Features

- **Real-time Data Processing**: Continuous monitoring of XAUUSD market data
- **Technical Analysis**: Advanced technical indicators using TA-Lib
- **Signal Generation**: Automated buy/sell signal generation based on multiple indicators
- **High Performance**: Async/await architecture with FastAPI for optimal performance
- **Redis Caching**: Fast data access and caching layer
- **Clean Architecture**: Domain-driven design with clear separation of concerns
- **Production Ready**: Docker containerization with health checks and monitoring
- **Comprehensive Testing**: 90%+ test coverage with pytest

## Architecture

The project follows clean architecture principles with three distinct layers: