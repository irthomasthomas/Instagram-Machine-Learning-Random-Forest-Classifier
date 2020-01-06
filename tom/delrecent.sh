for k in $(redis-cli keys "scraped:recent*"); do
  echo "delete key '$k'";
  redis-cli DEL $k;
done
