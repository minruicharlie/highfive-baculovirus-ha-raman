# 03 Raman Preprocessing

## Spectral Window and Feature Grid

```text
procedure SELECT_RAMAN_WINDOW(spectra):
    keep wavenumbers within the predefined fingerprint window
    optionally thin adjacent variables using a predefined stride
    return windowed_spectra
end procedure
```

## Chemical Preprocessing

```text
procedure PREPROCESS_RAMAN(raw_spectra, preprocessing_settings):
    for each spectrum:
        fit low-degree polynomial baseline across the selected wavenumber axis
        subtract fitted baseline from raw intensity

    apply Savitzky-Golay smoothing or derivative transformation

    for each transformed spectrum:
        center by its own mean intensity
        scale by its own standard deviation

    return preprocessed_spectra
end procedure
```

## Self-Reference Transformation

Some HA models use an early-run self-reference to reduce batch-level spectral offsets.

```text
procedure SELF_REFERENCE_DELTA(raw_spectra, batch_id, culture_time, reference_length):
    for each batch:
        order spectra by culture_time
        reference_spectra = first reference_length spectra in that batch
        batch_reference = median spectrum of reference_spectra

        for each spectrum in batch:
            delta_spectrum = spectrum - batch_reference

    return delta_spectra
end procedure
```

The reference calculation is performed within the allowed training context during cross-validation and then applied according to the locked route during final fitting.
