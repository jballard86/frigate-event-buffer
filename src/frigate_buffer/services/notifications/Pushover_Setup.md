# Pushover Notification Provider

This document explains how to enable and configure the Pushover notification provider for the Frigate Event Buffer.

## Prerequisites

1. **Pushover account**: Sign up at [pushover.net](https://pushover.net).
2. **User key**: In the [Pushover dashboard](https://pushover.net/dashboard), copy your User Key.
3. **Application / API token**: [Create an application](https://pushover.net/apps/build) to get an API token. Each app has its own token; use it in the `pushover_api_token` config option.

## Configurable options

Add a `pushover` key under the `notifications` block in your `config.yaml`. The provider is only enabled when `enabled` is true and both `pushover_api_token` and `pushover_user_key` are set (from config or environment).

| Option               | Type   | Default | Description |
|----------------------|--------|---------|-------------|
| `enabled`            | bool   | false   | When true and credentials are set, the orchestrator adds the Pushover provider. |
| `pushover_user_key`  | string | —       | Your Pushover User Key (required for sending). |
| `pushover_api_token` | string | —       | Your application's API token (required for sending). |
| `device`             | string | —       | Optional. Target device name(s); comma-separated for multiple devices. If omitted, all devices receive the notification. |
| `default_sound`      | string | —       | Optional. Override the default notification sound (e.g. `pushover`, `none`, `siren`, `vibrate`). See [Pushover sounds](https://pushover.net/api#sounds). |
| `html`               | int    | 1       | Set to `1` to enable HTML in the message body (e.g. `<b>bold</b>`). Set to `0` for plain text. |

## Environment variables

Credentials can be supplied via environment variables (e.g. in a `.env` file or your deployment environment). They override values from `config.yaml`.

| Variable             | Description |
|----------------------|-------------|
| `PUSHOVER_USER_KEY`  | Pushover User Key. Overrides `pushover.pushover_user_key` from config. |
| `PUSHOVER_API_TOKEN` | Pushover application API token. Overrides `pushover.pushover_api_token` from config. |

Example `.env`:

```env
PUSHOVER_USER_KEY=your_user_key_here
PUSHOVER_API_TOKEN=your_app_api_token_here
```

With env vars set, you can enable Pushover in config without putting secrets in YAML:

```yaml
notifications:
  pushover:
    enabled: true
    # user key and api token come from PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN
    device: "phone"           # optional
    default_sound: "pushover" # optional
    html: 1
```

## Behavior summary

- **Phase filter**: Only these statuses trigger a Pushover notification: `snapshot_ready`, `clip_ready`, and `finalized`. All other statuses (e.g. `new`, `described`) are skipped with a "Filtered intermediate phase" result.
- **snapshot_ready**: Normal priority (0), or high priority (1) when `threat_level` is high/critical (≥ 2). If `latest.jpg` exists in the event folder, it is attached.
- **clip_ready** / **finalized**: Low priority (-1), so the update is silent (no sound/vibration). If `notification.gif` exists in the event folder, it is attached.
- **Overflow**: When the notification queue overflows, the provider sends a single message: "Too many events occurring. Notifications temporarily paused." with normal priority.

## Example config (YAML only)

```yaml
notifications:
  pushover:
    enabled: true
    pushover_user_key: "YOUR_USER_KEY"
    pushover_api_token: "YOUR_APP_API_TOKEN"
    device: "phone"           # optional
    default_sound: "pushover" # optional
    html: 1
```

## Links

- [Pushover API](https://pushover.net/api)
- [Pushover dashboard](https://pushover.net/dashboard)
- [Create an application](https://pushover.net/apps/build)
