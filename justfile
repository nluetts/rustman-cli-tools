debugger:
	gdbgui

test-release:
	target/release/raman-cli-tools test/test_frames.csv reshape 1340 finning 2.0 integrate 660,661 > /dev/null
	target/release/raman-cli-tools test/test_frames.csv reshape 1340 finning 4.0 align > /dev/null

copy-ruman:
	cp ~/Repos/rustman-cli-tools/target/release/raman-cli-tools ~/.local/bin/ruman