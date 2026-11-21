# S1.7 checkpoint 2b toolchain image (Amendment 9). Base: the python:3.10-slim digest of the
# DS-1000 image (the sandbox boot program needs python3). Network only at build time; cells run
# with --network none. Built for linux/arm64; referenced by local image id (IMAGE_ID).
# CodeGeeX HumanEval-X Java: javac Main.java, then java -cp . Main (the run line is commented
# out in CodeGeeX's execution.py at 2838420; the intended command is used).
FROM eclipse-temurin:21.0.4_7-jdk@sha256:49deb4332bb815ce4792c3673936042711b5fe64c75752ff505b591768499c4f AS jdk
FROM python:3.10-slim@sha256:6f16c98fd47a233f0195fb4e683539a86211c2cea0ba448966612b38789596c9
COPY --from=jdk /opt/java/openjdk /opt/java/openjdk
RUN /opt/java/openjdk/bin/java -version 2> /opt/toolchain-versions.txt
