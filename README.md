# AGR Automated Information Extraction Models and Pipelines

This is the main Alliance repository for automated information extraction pipelines and models.

- Classify Alliance references into biocuration topics using machine learning models.
- Extract entities from references

The repository contains code to train models and upload them to the ABC. When used to classify new documents, the classifier related to the specified MOD abbreviation and topic (data type) is fetched from the ABC. 
Documents for training and classification are fetched from the ABC repository in TEI format. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Running a stage/test image](#running-a-stagetest-image)
- [Logging to Graylog](#logging-to-graylog)
- [Development](#development)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/agr_automated_information_extraction.git
    cd agr_automated_information_extraction
    ```

2. Create and configure the `.env` file:
    ```sh
    cp .env.example .env
    # Edit the .env file to include your specific configuration
    ```
   
3. Build the Docker image:
    ```sh
    docker-compose build
    ```


## Usage

### Training a Classifier

To train a classifier, run the following command:
```sh
docker-compose run agr_document_classifier python agr_document_classifier.py --mode train --datatype_train <topic_ATP_ID> --mod_train <mod_abbreviation> --embedding_model_path <path_to_embedding_model>
```

### Optional Arguments for training

- --weighted_average_word_embedding: Use weighted average for word embeddings.
- --standardize_embeddings: Standardize the embeddings.
- --normalize_embeddings: Normalize the embeddings.
- --sections_to_use: Specify sections to use for training.
- --skip_training_set_download: Skip downloading the training set.
- --skip_training: Skip the training process and upload a pre-existing model.


### Classifying Documents

To classify documents, run the following command:
```sh
docker-compose run agr_document_classifier python agr_document_classifier.py --mode classify --embedding_model_path <path_to_embedding_model>
```

### Configuration

The project uses environment variables for configuration. These variables are defined in the .env file. Key variables include:  

- TRAINING_DIR: Directory for training data.
- CLASSIFICATION_DIR: Directory for documents to classify.
- CLASSIFIERS_PATH: Path to save classifiers.
- ABC_API_SERVER: URL for the ABC API server.
- OKTA_*: Configuration for Okta authentication.
- CLASSIFICATION_BATCH_SIZE: Batch size for document classification.
- ENV_STATE / GELF_ADDRESS: Centralized logging, see below.

## Running a stage/test image

The stage image is built independently — normally from a feature branch — and runs
against the stage/test database. It is a separate build, not a retag of the prod
image, so a later prod build cannot silently replace it.

```sh
# build it (tags agr.literature.automated_information_extraction.server-test)
make doc_classifier_build_test

# run it on the same host as prod, pointed at the stage DB via a different env file
make classify ENV_FILE=.env.stage \
  APP_IMAGE=agr.literature.automated_information_extraction.server-test
```

`APP_IMAGE` selects which image the compose service runs and defaults to the prod
image. Database credentials and every other setting come from `ENV_FILE`, so prod
and stage can run side by side on one machine without interfering. Set
`ENV_STATE=stage` in that env file to keep the two apart in Graylog.

Note GoCD builds these images itself with raw `docker build --no-cache`, not via
make, so the tags it applies are configured in the pipeline rather than here.

## Logging to Graylog

Container stdout and stderr are shipped to the Alliance Graylog instance at
`udp://agr-log-nlb-prod-2338fbd566b1dd01.elb.us-east-1.amazonaws.com:12201`
using Docker's built-in `gelf` log driver,
the same mechanism `agr_literature_service` uses. There is no Python-side GELF
library and no application code involved — anything a pipeline writes to stdout or
stderr is forwarded by the Docker daemon.

The driver is declared in the `logging:` block of `docker-compose.yaml` and is
controlled by three variables, all consumed by Compose during interpolation rather
than by the container process (so they must **not** be added to the `environment:`
list):

| Variable | Default | Purpose |
| --- | --- | --- |
| `GELF_ADDRESS` | `udp://agr-log-nlb-prod-2338fbd566b1dd01.elb.us-east-1.amazonaws.com:12201` | Where to send GELF packets. |
| `ENV_STATE` | `dev` | Separates dev / stage / prod streams. Set to `prod` on production hosts. |
| `GELF_COMPONENT` | `unspecified` | Which pipeline is running. Set per invocation by the Makefile targets. |

### Finding your run in Graylog

Search on the `tag` field, which is built as `agr.aie.<ENV_STATE>.<GELF_COMPONENT>` —
for example `tag:agr.aie.prod.classify`. The Docker driver also attaches
`container_name`, `container_id`, `image_name`, `command` (the exact `python <script>
--args` line) and `host`, so individual runs remain distinguishable even without a tag.

### Jobs launched outside Compose

The FlyBase textmining jobs run from GoCD as bare `docker run`, which bypasses
`docker-compose.yaml`. They need the equivalent flags on the `docker run` command
itself (see the header comments in `bin/run_export_and_commit.sh` and
`bin/check_textmining_freshness.sh`):

```sh
docker run --rm \
  --log-driver gelf \
  --log-opt gelf-address=udp://agr-log-nlb-prod-2338fbd566b1dd01.elb.us-east-1.amazonaws.com:12201 \
  --log-opt tag=agr.aie.prod.fb_textmining_export \
  ...
```

### Network requirement: the host must be inside the Alliance VPC

`agr-log-nlb-prod-2338fbd566b1dd01.elb.us-east-1.amazonaws.com` resolves to
`172.31.96.173`, an RFC1918 private address in the Alliance VPC (it is an
internal NLB). **Only hosts inside that network can deliver GELF packets.**
Public DNS resolves the name from anywhere, so resolution succeeding tells you
nothing about reachability.

This fails silently and is easy to misdiagnose:

- the name resolves, so the container starts and the driver initialises normally;
- GELF over UDP is fire-and-forget — no connection, no ACK, no error;
- the job runs to completion and exits 0, output appears on the console as usual,
  and the messages simply never arrive in Graylog.

Confirmed unreachable from the FlyBase GoCD agent `flysql26` (100% packet loss to
the Graylog input's private address). Checks, run on the host in question:

```sh
ip route get 172.31.96.173     # default gateway means no path into the VPC
ping -c2 172.31.96.173
traceroute -n -U -p 12201 172.31.96.173
```

For hosts outside the VPC, either point `GELF_ADDRESS` at a reachable endpoint or
omit the `--log-driver`/`--log-opt` flags entirely — for those jobs the GoCD console
remains the log of record. Getting them into Graylog is a networking change
(a reachable GELF input, or peering/firewall access), not a change to this repo.

### Caveats

- **`docker logs` no longer works.** With a non-`json-file` driver Docker keeps no
  local copy. Interactive `docker-compose run` still prints to your terminal, since
  attach is independent of the log driver, so day-to-day development is unaffected.
- **The container will not start if the GELF address cannot be resolved.** Docker
  resolves it at container-create time. To opt out locally, create an uncommitted
  `docker-compose.override.yaml` (Compose loads it automatically):
  ```yaml
  services:
    agr_automated_information_extraction:
      logging:
        driver: json-file
  ```
- **Only stdout/stderr is shipped.** Scripts that redirect output to a file (the
  `crontab` entry, the `.log` files parsed by
  `scripts/fb_gene_extraction_summarize_run_logs.py`) bypass Graylog entirely.
- **GELF over UDP is lossy and unordered**, and messages larger than ~8KB are
  chunked. Do not rely on Graylog for anything that must not be dropped.
