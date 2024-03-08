for k in $(redis-cli keys "/enqueue*"); do
  echo "delete key '$k'";
  redis-cli DEL $k;
done
