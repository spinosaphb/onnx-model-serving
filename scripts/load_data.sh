#!/bin/bash

export KAGGLE_USERNAME="pedrospinosa"
export KAGGLE_KEY="2b39535cb999bc74841fdfe408aca210"

COMPETITION="dogs-vs-cats"

DESTINATION_DIR="workspace/datasets/$COMPETITION"

mkdir -p "$DESTINATION_DIR"

kaggle competitions download -c "$COMPETITION" -p "$DESTINATION_DIR"

unzip "$DESTINATION_DIR/$COMPETITION.zip" -d "$DESTINATION_DIR"

rm "$DESTINATION_DIR/$COMPETITION.zip"