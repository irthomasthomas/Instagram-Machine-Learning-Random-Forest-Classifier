#!/bin/bash
echo "deleting...";
for k in $(redis-cli keys "tag:posts:*"); do
  echo "delete key '$k'";
  redis-cli DEL $k;
done
