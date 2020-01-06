for k in $(redis-cli keys "tags:out*"); do
  echo "delete key '$k'";
  redis-cli DEL $k;
done
