import math
import shutil

DOMAINSPERWET = 20000
def get_extracted_wet(spark_context, approx_sample_size, wet_paths_file = "data/wet.paths", seed = 0):
    sc = spark_context
    wets_to_load = math.ceil(approx_sample_size / DOMAINSPERWET)
    paths_rdd = sc.textFile(wet_paths_file)
    sampled_paths = sc.parallelize(paths_rdd.takeSample(withReplacement=False, num=wets_to_load, seed=seed))
    url_head = "https://data.commoncrawl.org/"
    urls_rdd = sampled_paths.map(lambda x: url_head + x)

    # we use requests to get a http response from the common crawl server
    def get_request(url):
        import requests
        import random
        import time

        jitter = random.random()
        time.sleep(jitter)
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            return response
        else:
            # Here we implement an exponential back off strategy to make sure we're not overloading the server
            # Here we wait for a total of (2^11)/10 = 204.8 seconds or about 3.5 minutes max to get a response from the server
            # So total wait time can be a bit over 5 minutes
            for i in range(12):
                jitter = random.random()
                time.sleep((2 ** i + jitter) / 10)
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    # Request was successful
                    return response
            return None

            # This is to have our responses RDD be filtered so we don't have any Nonetypes

    responses_rdd = urls_rdd.map(get_request).filter(lambda x: x is not None)

    # we convert the responses into gzip files in memory
    def get_zipped(response):
        import io
        return io.BytesIO(response.content)

    zipped_files = responses_rdd.map(get_zipped)

    # we unzip the zipped files also within memory
    def unzip(zipped):
        import gzip
        with gzip.GzipFile(fileobj=zipped) as decompressed_file:
            return decompressed_file.read()

    unzipped_rdd = zipped_files.map(unzip).persist()

    str_rdd = unzipped_rdd.map(lambda x: x.decode("utf-8"))

    return str_rdd

def save_rdd(rdd, path_to_save, overwrite=False):
    if overwrite:
        shutil.rmtree(path_to_save)
    rdd.saveAsPickleFile(path_to_save)

def load_rdd(spark_context, path_to_load):
    return spark_context.pickleFile(path_to_load)