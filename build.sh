#!/bin/sh

PROJECT_VERSION=$(git rev-parse --short HEAD) cargo build --release
