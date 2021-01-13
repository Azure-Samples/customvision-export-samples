# Sample script for CustomVision's OpenVino Classification model

| Task | Domain | Export Platform | Export Flavor |
|------|--------|-----------------|---------------|
| Classification | General (compact) [S1] | OpenVino | null |
| Classification | General (compact) [S1] | OpenVino | NoPostProcess |
| Classification | General (compact) | OpenVino | null |
| Classification | General (compact) | OpenVino | NoPostProcess |

For the detail of the model export features, please visit [Custom Vision's official documents](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/).

## Setup
- Follow [README](../README.md) to set up OpenVino environment.

## How to use
```
python predict.py <xml_filepath> <bin_filepath> <image_filepath>
```

## Notes
There is a slight difference in image preprocessing logic and this script cannot get identical results with Custom Vision's Cloud API.
