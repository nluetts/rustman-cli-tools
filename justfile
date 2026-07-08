debugger:
	gdbgui

vibe:
	repovibe.py --exclude Cargo.lock,./target/*,*.csv . /tmp/digest.md

test-release:
	target/release/raman-cli-tools test/test_frames.csv reshape 1340 finning 2.0 integrate 660,661 > /dev/null
	target/release/raman-cli-tools test/test_frames.csv reshape 1340 finning 4.0 align > /dev/null

copy-ruman:
	cp ~/Repos/rustman-cli-tools/target/release/raman-cli-tools ~/.local/bin/ruman

build:
	toolbox run -c dflt env PROJECT_VERSION=$(git rev-parse --short HEAD) cargo build --release

build-win:
	# toolbox run -c dflt env PROJECT_VERSION=$(git rev-parse --short HEAD) cargo build --release --target x86_64-pc-windows-gnu
	toolbox run -c dflt env PROJECT_VERSION=$(git rev-parse --short HEAD) RUSTFLAGS="-L /usr/x86_64-w64-mingw32/lib" MAKEFLAGS="-j$(nproc)" OPENBLAS_TARGET=HASWELL OPENBLAS_NUM_THREADS=$(nproc) cargo build --release --target x86_64-pc-windows-gnu

release:
	just build
	cp ./target/release/raman-cli-tools ~/.local/bin/ruman

release-win:
	toolbox run -c dflt env PROJECT_VERSION=$(git rev-parse --short HEAD) cargo build --release --target x86_64-pc-windows-gnu
	cp ./target/x86_64-pc-windows-gnu/release/raman-cli-tools.exe ~/ownCloud/Exchange/DS_Exchange/Software/ruman.exe
