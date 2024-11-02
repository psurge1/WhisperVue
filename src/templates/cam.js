export default function VidStream({altLabel}) {

    return (
      <div>
        <img
          src="{{ url_for('video_feed') }}"
          alt={altLabel}
        />
      </div>
    )
}