/**
 * Guard for camera/microphone access.
 *
 * WHY THIS EXISTS
 *
 * On an insecure origin browsers do not merely *deny* getUserMedia — they do
 * not define `navigator.mediaDevices` at all. Calling straight through gives:
 *
 *     Cannot read properties of undefined (reading 'getUserMedia')
 *
 * which tells a user nothing, and is especially bad here because the person
 * hitting it may not be able to see the screen. The actual cause is almost
 * always "opened over http:// from a phone", and the actual fix is one URL
 * change — so say that, and say it with the exact address to use.
 *
 * A secure context means HTTPS, or localhost. A LAN address over plain HTTP is
 * neither, which is the whole reason this project ships a certificate script.
 */

export class MediaUnavailableError extends Error {
  constructor(message: string, readonly hint: string, readonly suggestedUrl?: string) {
    super(message);
    this.name = 'MediaUnavailableError';
  }
}

/** The HTTPS address of this same page, for the "open this instead" hint. */
export function secureUrlForThisPage(): string {
  const { hostname, port, pathname } = window.location;
  return 'https://' + hostname + (port ? ':' + port : '') + pathname;
}

export function isLocalhost(): boolean {
  const h = window.location.hostname;
  return h === 'localhost' || h === '127.0.0.1' || h === '::1' || h === '[::1]';
}

/**
 * Throws a MediaUnavailableError explaining precisely what is wrong, or
 * returns the MediaDevices object when capture is genuinely possible.
 */
export function requireMediaDevices(): MediaDevices {
  if (typeof navigator === 'undefined') {
    throw new MediaUnavailableError(
      'No browser environment available.',
      'This page must run in a browser.',
    );
  }

  if (!window.isSecureContext && !isLocalhost()) {
    throw new MediaUnavailableError(
      'The camera and microphone need a secure connection.',
      'You opened this over http://, and browsers only allow camera access on ' +
        'https:// or localhost. Open the https:// address instead and accept the ' +
        'certificate warning once.',
      secureUrlForThisPage(),
    );
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    // Secure context but still missing: an old browser, or an embedded webview
    // that withholds the API.
    throw new MediaUnavailableError(
      'This browser will not give access to the camera or microphone.',
      'Try Chrome, Edge, or Safari directly rather than an in-app browser ' +
        '(the one inside Instagram, WhatsApp and similar apps blocks capture).',
    );
  }

  return navigator.mediaDevices;
}

/** A single sentence safe to show, and to read aloud, for any capture failure. */
export function describeMediaError(error: unknown): string {
  if (error instanceof MediaUnavailableError) {
    return error.suggestedUrl
      ? error.message + ' ' + error.hint + ' Address: ' + error.suggestedUrl
      : error.message + ' ' + error.hint;
  }
  if (error instanceof DOMException) {
    switch (error.name) {
      case 'NotAllowedError':
        return 'Permission was denied. Allow camera and microphone access for this site, then press Start again.';
      case 'NotFoundError':
        return 'No camera or microphone was found on this device.';
      case 'NotReadableError':
        return 'The camera is already in use by another app. Close that app and try again.';
      case 'OverconstrainedError':
        return 'This device cannot provide the requested camera settings.';
      default:
        return 'Could not start capture: ' + error.name + '.';
    }
  }
  return error instanceof Error ? error.message : 'Could not start capture.';
}
