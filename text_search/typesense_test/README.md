# Typesense Text Search

This is a simple implementation of typesense. It includes a script to index the data
and a class designed to search the data given a question. We have an implementation
that includes the question classifier as well.

Typesense, since I chose not to use the cloud version, is a self-hosted search engine.
It is pretty easy to use, but does require some setup. I would recommend using the
docker installation, since it is the easiest to get up and running. Here is their
official guide to getting started: https://typesense.org/docs/guide/install-typesense.html

After installation, run the following command to start typesense. Just be sure to change
the directory you installed to, api key, or port as needed.

```bash
docker run -p 8108:8108 \
            -v/home/msaad/typesense-data:/data typesense/typesense:0.25.2 \
            --data-dir /data \
            --api-key=xyz \
            --enable-cors
```

