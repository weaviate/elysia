# chunk a collection
# 1. perform an initial query to get X relevant objects (X > 100 probably)
# 2. iterate through the objects, chunk the 'content' field
# 3. save the chunks to a new collection
# 4. new collection has references to original
# 5. the objects in new collection are only indices of chunk position, doc_id of original object, and quantized vectors of the chunk
# 6. the collection is cached, and when chunking on-demand, we first check to see if this object exists in the cache

