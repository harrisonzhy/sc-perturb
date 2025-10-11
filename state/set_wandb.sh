#!/bin/bash

entity="harrisonzhy"
sed -i "s|entity: your_entity_name|entity: ${entity}|g" src/state/configs/wandb/default.yaml

