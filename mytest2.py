import requests
import sys
vs2015_link = 'https://docs.google.com/uc?export=download&confirm=xiw_&id=1eH5tCm1GL3jLvSquIOEulHDSJ7ErzMJN'

file_name = "Visual Studio 2015 Professional.zip"
with open(file_name, "wb") as f:
        print("Downloading %s" % file_name)
        response = requests.get(vs2015_link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None: # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )
                sys.stdout.flush()