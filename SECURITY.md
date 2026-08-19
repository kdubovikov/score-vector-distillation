# Security

Do not report credentials in a public issue. Use GitHub private vulnerability
reporting for suspected secret exposure or unsafe model-loading behavior.

The release audit rejects credential files, private paths, known internal cluster or scheduler markers, and common secret formats. Configurations requiring `trust_remote_code` fail closed until an immutable model revision is pinned; review that remote code before resolving the pin.
